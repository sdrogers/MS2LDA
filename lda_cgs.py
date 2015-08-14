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

"""

import cPickle
import sys
import time
from collections import namedtuple

from numpy import int32
from numpy.random import RandomState
import gzip

from lda_generate_data import LdaDataGenerator
import lda_utils as utils
import numpy as np
import pandas as pd
import pylab as plt
from scipy.special import psi
from lda_utils import estimate_alpha_from_counts
import visualisation.pyLDAvis as pyLDAvis

Sample = namedtuple('Sample', 'cdk ckn')

class CollapseGibbsLda(object):
    
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
                N_prev_words = len(self.previous_vocab)
                N_diff = self.N - N_prev_words
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
                
                # initialise the Dirichlet hyperparameters too
                self.alpha = np.ones(self.K) * alpha
                self.beta = np.ones(self.N) * beta
                for n in range(N_prev_words):
                    # but make sure to use previous beta for the words that have been fixed from before
                    self.beta[n] = self.previous_model.selected_beta[n] 
                
            else:                
                raise ValueError("No previous topics have been selected")
                
        else:

            # for training stage
            self.K = K            
            self.previous_ckn = np.zeros((self.K, self.N), int32)
            self.previous_ck = np.zeros(self.K, int32)
            self.previous_K = 0 # no old topics
            self.alpha = np.ones(self.K) * alpha
            self.beta = np.ones(self.N) * beta
            

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
            
        self.samples = [] # store the samples
        
    def store_sample(self):
        cdk_copy = np.copy(self.cdk)
        ckn_copy = np.copy(self.ckn)
        samp = Sample(cdk_copy, ckn_copy)
        self.samples.append(samp)

    def _get_posterior_probs(self, samp_cdk, samp_ckn):

        # update theta
        theta = samp_cdk + self.alpha 
        theta /= np.sum(theta, axis=1)[:, np.newaxis]
        
        # update phi
        phi = samp_ckn + self.beta
        phi /= np.sum(phi, axis=1)[:, np.newaxis]
        
        # update posterior alpha
        alpha_new = estimate_alpha_from_counts(self.D, self.K, self.alpha, samp_cdk)      
        return theta, phi, alpha_new
            
    def _update_parameters(self):

        # use the last sample only
        if self.n_burn == 0:
            print "Using only the last sample"
            last_samp = self.samples[-1]
            theta, phi, alpha_new = self._get_posterior_probs(last_samp.cdk, last_samp.ckn)            
            return phi, theta, alpha_new

        print "Using all samples"
        thetas = []
        phis = []
        alphas = []
        for samp in self.samples:            
            theta, phi, alpha_new = self._get_posterior_probs(samp.cdk, samp.ckn)
            thetas.append(theta)
            phis.append(phi)
            if not np.isnan(alpha_new).any():           
                alphas.append(alpha_new)
        
        # average over the results
        S = len(self.samples)
        avg_theta = np.zeros_like(thetas[0])
        avg_phi = np.zeros_like(phis[0])
        avg_posterior_alpha = np.zeros_like(alphas[0])

        for theta in thetas:
            avg_theta += theta
        avg_theta /= len(thetas)
        
        for phi in phis:
            avg_phi += phi
        avg_phi /= len(phis)
        
        for alpha in alphas:
            avg_posterior_alpha += alpha
        avg_posterior_alpha /= len(alphas)
        
        return avg_phi, avg_theta, avg_posterior_alpha
                                                    
    def run(self, n_burn, n_samples, n_thin=1, use_native=True):
        
        self.n_burn = n_burn
        self.n_thin = n_thin
        if self.n_burn == 0:
            self.n_thin = 1
        
        """ 
        Runs the Gibbs sampling for LDA 
        
        Arguments:
        - n_burn: no of initial burn-in samples
        - n_samples: no of samples, must be > n_burn
        - n_thin: how often to thin the log_likelihood results stored
        - use_native: if True, will call the sampling function in lda_cgs_numba.py
        """

        # select the sampler function to use
        from lda_cgs_numpy import sample_numpy
        sampler_func = None
        if not use_native:
            print "Using Numpy for LDA sampling"
            sampler_func = sample_numpy
        else:
            print "Using Numba for LDA sampling"
            try:
                from lda_cgs_numba import sample_numba
                sampler_func = sample_numba
            except Exception:
                print "Numba not found. Using Numpy for LDA sampling"
                sampler_func = sample_numpy            

        # this will modify the various count matrices (Z, cdk, ckn, cd, ck) inside
        self.loglikelihoods_, self.samples = sampler_func(
                self.random_state, n_burn, n_samples, n_thin,
                self.D, self.N, self.K, self.document_indices,
                self.alpha, self.beta,
                self.Z, self.cdk, self.cd, self.previous_K,
                self.ckn, self.ck, self.previous_ckn, self.previous_ck)
        
        # update posterior alpha from the last sample  
        self.topic_word_, self.doc_topic_, self.posterior_alpha = self._update_parameters()   
                        
    @classmethod
    def load(cls, filename):
        with gzip.GzipFile(filename, 'rb') as f:
            obj = cPickle.load(f)
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
        
        # and the beta hyperparameter for those words too
        self.selected_beta = self.beta[non_zero_cols_pos]

        # dump the whole model out
        # binary mode ('b') is required for portability between Unix and Windows
        with gzip.GzipFile(model_out, 'wb') as f:
            cPickle.dump(self, f, protocol=cPickle.HIGHEST_PROTOCOL)
            print "Model saved to " + model_out

        # also write out the selected vocabs into a text file
        # can be used for feature processing later ..
        with open(words_out, 'w') as f:
            for item in self.selected_vocab:
                f.write("{}\n".format(item))                
        print "Words written to " + words_out    
        
    def visualise(self, topic_plotter):
        data = {}
        data['topic_term_dists'] = self.topic_word_
        data['doc_topic_dists'] = self.doc_topic_
        data['doc_lengths'] = self.cd
        data['vocab'] = self.vocab
        data['term_frequency'] = np.sum(self.ckn, axis=0)    
        data['topic_ranking'] = topic_plotter.topic_ranking
        data['topic_coordinates'] = topic_plotter.topic_coordinates
        data['plot_opts'] = {'xlab': 'h-index', 'ylab': 'log(degree)', 'sort_by' : topic_plotter.sort_by}
        data['lambda_step'] = 5         
        data['lambda_min'] = self._round_nicely(topic_plotter.sort_by_min)
        data['lambda_max'] = self._round_nicely(topic_plotter.sort_by_max)
        vis_data = pyLDAvis.prepare(**data)   
        pyLDAvis.show(vis_data, topic_plotter=topic_plotter)   

    # http://stackoverflow.com/questions/2272149/round-to-5-or-other-number-in-python        
    def _round_nicely(self, x, base=5):
        return int(base * round(float(x)/base))             
        
    def threshold_matrix(self, matrix, epsilon=0.0):
        thresholded = matrix.copy()
        n_row, n_col = thresholded.shape
        for i in range(n_row):
            row = thresholded[i, :]
            if epsilon > 0:
                small = row < epsilon
                row[small] = 0
            else:
                smallest_val = np.min(row)
                smallest_arr = np.ones_like(row) * smallest_val
                close = np.isclose(row, smallest_arr)
                row[close] = 0        
        return thresholded
        
def main():

    multiplier = 1
    n_topics = 20 * multiplier
    n_docs = 100 * multiplier
    vocab_size = 200 * multiplier
    document_length = 50 * multiplier

    alpha = 0.1
    beta = 0.01    
    n_samples = 400
    n_burn = 200
    n_thin = 10

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
    gibbs1.print_topic_words()
      
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