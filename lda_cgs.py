"""
An implementation of a collapsed Gibbs sampling for Latent Dirichlet Allocation [1]

Two variant implementations of the Gibbs sampling method are provided:

- lda_cgs_numpy.py contains a Numpy implementation of the collapsed Gibbs sampling

- lda_cgs_numba.py contains a version of the sampler written in (only) Python, but structured in
  such a way that it can be accelerated by Numba [2], a JIT translator of Python bytecode to LLVM 
  (http://numba.pydata.org). This mostly entails converting vectorised Numpy operations into unrolled 
  loops, resulting in some performance boost (5x or more) in the compiled code over Numpy.
  However, it is still slower than the LDA version found at https://pypi.python.org/pypi/lda, 
  written in Cython.

[1] Blei, David M., Andrew Y. Ng, and Michael I. Jordan. "Latent dirichlet allocation." 
the Journal of machine Learning research 3 (2003): 993-1022.

[2] Oliphant, Travis. "Numba python bytecode to LLVM translator." 
Proceedings of the Python for Scientific Computing Conference (SciPy). 2012.

The variational Bayes version of this file can also be found in lda_vb.py

"""

import cPickle
import sys
import time

from lda.lda import LDA
from numpy import int32
from numpy.random import RandomState

from lda_cgs_numba import sample_numba
from lda_cgs_numpy import sample_numpy
from lda_generate_data import LdaDataGenerator
import lda_utils as utils
import numpy as np
import pandas as pd
import pylab as plt
from scipy.special import psi
from lda_utils import estimate_alpha_from_theta


class CollapseGibbsLda:
    
    def __init__(self, df, vocab, K, alpha, beta, random_state=None, previous_model=None):
        """
        Initialises the collapsed Gibbs sampling for LDA
        
        Arguments:
        - df: the dataframe of counts of vocabularies x documents
        - K: no. of topics
        - alpha: symmetric prior on document-topic assignment
        - beta: symmetric prior on word-topic assignment
        - previous_model: previous LDA run, if any
        """
        
        print "CGS LDA initialising"
        self.df = df.replace(np.nan, 0)
        self.alpha = alpha
        self.beta = beta

        self.D = df.shape[0]    # total no of docs
        self.N = df.shape[1]    # total no of words
        self.vocab = vocab
        assert(len(self.vocab)==self.N)

        # set total no of topics
        self.cv = False
        self.previous_model = previous_model
        if self.previous_model is not None:
            
            # if some old topics were fixed
            if hasattr(self.previous_model, 'selected_topics'):
            
                # no. of new topics
                self.K = K
            
                # no. of previously selected topics
                self.previous_K = len(self.previous_model.selected_topics)
                
                # Get the previous ckn and ck values from the training stage.
                # During gibbs update in this testing stage, assignment of word 
                # to the first previous_K topics will use the previous fixed 
                # topic-word distributions -- as specified by previous_ckn and previous_ck
                self.previous_ckn = self.previous_model.selected_ckn
                self.previous_ck = self.previous_model.selected_ck
                self.previous_vocab = self.previous_model.selected_vocab
                assert(len(self.previous_ck)==self.previous_K)
                assert(self.previous_ckn.shape[0]==len(self.previous_ck))
                assert(self.previous_ckn.shape[1]==len(self.previous_vocab))
                
                # make previous_ckn have the right number of columns
                N_diff = self.N - len(self.previous_vocab)
                temp = np.zeros((self.previous_K, N_diff), int32)
                self.previous_ckn = np.hstack((self.previous_ckn, temp)) # size is previous_K x N
                
                # make previous_ckn have the right number of rows
                temp = np.zeros((self.K, self.N), int32)
                self.previous_ckn = np.vstack((self.previous_ckn, temp)) # size is (previous_K+K) x N

                # make previous_ck have the right length
                temp = np.zeros(self.K, int32)
                self.previous_ck = np.hstack((self.previous_ck, temp)) # length is (previous_K+K)

                # total no. of topics = old + new topics
                self.K = self.K + self.previous_K
                print "Total no. of topics = " + str(self.K)
                
                
            else:                
                raise ValueError("No previous topics have been selected")
                
        else:

            # for training stage
            self.K = K            
            self.previous_ckn = np.zeros((self.K, self.N), int32)
            self.previous_ck = np.zeros(self.K, int32)
            self.previous_K = 0 # no old topics

        # make the current arrays too
        self.ckn = np.zeros((self.K, self.N), int32)
        self.ck = np.zeros(self.K, int32)
        self.cdk = np.zeros((self.D, self.K), int32)
        self.cd = np.zeros(self.D, int32)

        # make sure to get the same results from running gibbs each time
        if random_state is None:
            self.random_state = RandomState(1234567890)
        else:
            self.random_state = random_state

        # randomly assign words to topics
        self.Z = {}        
        for d in range(self.D):
            if d%10==0:
                sys.stdout.write('.')
                sys.stdout.flush()
            document = self.df.iloc[[d]]
            word_idx = utils.word_indices(document)
            for pos, n in enumerate(word_idx):
                k = self.random_state.randint(self.K)
                self.cdk[d, k] += 1
                self.cd[d] += 1
                self.ckn[k, n] += 1
                self.ck[k] += 1
                self.Z[(d, pos)] = k
        print

        # turn word counts in the document into a vector of word occurences
        self.document_indices = {}
        for d in range(self.D):
            document = self.df.iloc[[d]]
            word_idx = utils.word_indices(document)
            word_locs = []
            for pos, n in enumerate(word_idx):
                word_locs.append((pos, n))
            self.document_indices[d] = word_locs
            
    def get_posterior_alpha(self, n_iter=100):
        """
        Estimate the concentration parameter alpha from the thetas in the last sample
        """
        alpha_new = estimate_alpha_from_theta(self.D, self.K, self.alpha, self.doc_topic_, n_iter=100)      
        return alpha_new
                                                    
    def run(self, n_burn, n_samples, n_thin, use_native=False):
        """ 
        Runs the Gibbs sampling for LDA 
        
        Arguments:
        - n_burn: no of initial burn-in samples
        - n_samples: no of samples, must be > n_burn
        - n_thin: how often to thin the log_likelihood results stored
        - use_native: if True, will call the sampling function in lda_cgs_numba.py
        """

        # select the sampler function to use
        sampler_func = None
        if not use_native:
            print "Using Numpy for LDA sampling"
            sampler_func = sample_numpy
        else:
            print "Using Numba for LDA sampling"
            sampler_func = sample_numba

        # this will modify the various count matrices (Z, cdk, ckn, cd, ck) inside
        self.topic_word_, self.doc_topic_, self.loglikelihoods_ = sampler_func(
                self.random_state, n_burn, n_samples, n_thin,
                self.D, self.N, self.K, self.document_indices,
                self.alpha, self.beta,
                self.Z, self.cdk, self.cd, self.previous_K,
                self.ckn, self.ck, self.previous_ckn, self.previous_ck)
        
        # update posterior alpha from the last sample  
        self.posterior_alpha = self.get_posterior_alpha()   
                        
    @classmethod
    def load(cls, filename):
        f = file(filename, 'rb')
        obj = cPickle.load(f)
        f.close()
        print "Model loaded from " + filename
        return obj
    
    def save(self, topic_indices, model_out, words_out):
        
        self.selected_topics = topic_indices

        # Gets the ckn and ck matrices, but only for the selected rows and 
        # with all-zero columns removed         
        self.selected_ckn = self.ckn[topic_indices, :].copy()
        self.selected_ck = self.ck[topic_indices].copy()
        colsum = np.sum(self.selected_ckn, axis=0)
        all_zero_cols = (colsum==0)
        all_zero_cols_pos = np.where(all_zero_cols)
        self.selected_ckn = np.delete(self.selected_ckn, all_zero_cols_pos, 1)

        # also save the words used by topic_indices
        non_zero_cols = (colsum>0)
        non_zero_cols_pos = np.where(non_zero_cols)
        self.selected_vocab = self.vocab[non_zero_cols_pos] 

        # dump the whole model out
        # binary mode ('b') is required for portability between Unix and Windows
        f = file(model_out, 'wb')
        cPickle.dump(self, f)
        f.close()
        print "Model saved to " + model_out

        # also write out the selected vocabs into a text file
        # can be used for feature processing later ..
        with open(words_out, 'w') as f:
            for item in self.selected_vocab:
                f.write("{}\n".format(item))                
        print "Words written to " + words_out            
                                    
def main():

    multiplier = 1
    n_topics = 20 * multiplier
    n_docs = 100 * multiplier
    vocab_size = 200 * multiplier
    document_length = 50 * multiplier

    alpha = 0.1
    beta = 0.01    
    n_samples = 200
    n_burn = 0
    n_thin = 1

    random_state = RandomState(1234567890)

    gen = LdaDataGenerator(alpha, make_plot=True)
#     df, vocab = gen.generate_input_df(n_topics, vocab_size, document_length, n_docs, 
#                                       previous_vocab=None, vocab_prefix='gibbs1', 
#                                       df_outfile='input/test1.csv', vocab_outfile='input/test1.words')
    df, vocab = gen.generate_from_file('input/test1.csv', 'input/test1.words')

    gibbs1 = CollapseGibbsLda(df, vocab, n_topics, alpha, beta, random_state=random_state, previous_model=None)
    start_time = time.time()
    gibbs1.run(n_burn, n_samples, n_thin, use_native=True)
    print("--- TOTAL TIME %d seconds ---" % (time.time() - start_time))
    print gibbs1.posterior_alpha
      
#     # try saving model
#     selected_topics = [0, 2, 4, 6, 8]
#     gibbs1.save(selected_topics, 'input/gibbs1.p', 'input/gibbs1.selected.words')
#      
#     # try loading model
#     gibbs1 = CollapseGibbsLda.load('input/gibbs1.p')
#     if hasattr(gibbs1, 'selected_topics'):
#         print "Kept topics = " + str(gibbs1.selected_topics)
#    
#     gen._plot_nicely(gibbs1.doc_topic_.T, 'Inferred Topics X Docs', 'docs', 'topics', outfile='test1_doc_topic.png')
#     gen._plot_nicely(gibbs1.topic_word_, 'Inferred Topics X Terms', 'terms', 'topics', outfile='test1_topic_word.png')
#     plt.plot(gibbs1.loglikelihoods_)
#     plt.show()
#     
#     topic_word = gibbs1.topic_word_
#     n_top_words = 20
#     for i, topic_dist in enumerate(topic_word):
#         topic_words = vocab[np.argsort(topic_dist)][:-n_top_words:-1]
#         print('Topic {}: {}'.format(i, ' '.join(topic_words)))
#             
#     # now run gibbs again on another df with the few selected topics above
#     gen = LdaDataGenerator(alpha, make_plot=True)
# #     df2, vocab2 = gen.generate_input_df(n_topics, vocab_size, document_length, n_docs, 
# #                                         previous_vocab=gibbs1.selected_vocab, vocab_prefix='gibbs2', 
# #                                         df_outfile='input/test2.csv', vocab_outfile='input/test2.words')
#     df2, vocab2 = gen.generate_from_file('input/test2.csv', 'input/test2.words')
#     gibbs2 = CollapseGibbsLda(df2, vocab2, n_topics, alpha, beta, previous_model=gibbs1)
#     gibbs2.run(n_burn, n_samples, n_thin, use_native=True)
#       
#     gen._plot_nicely(gibbs2.doc_topic_.T, 'Inferred Topics X Docs', 'docs', 'topics', outfile='test2_doc_topic.png')
#     gen._plot_nicely(gibbs2.topic_word_, 'Inferred Topics X Terms', 'terms', 'topics', outfile='test2_topic_word.png')
#     plt.plot(gibbs2.loglikelihoods_)
#     plt.show()
#        
#     topic_word = gibbs2.topic_word_
#     n_top_words = 20
#     for i, topic_dist in enumerate(topic_word):
#         topic_words = vocab2[np.argsort(topic_dist)][:-n_top_words:-1]
#         print('Topic {}: {}'.format(i, ' '.join(topic_words)))

if __name__ == "__main__":
    main()