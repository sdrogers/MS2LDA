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

import sys
import time

from lda.lda import LDA
from numpy.random import RandomState

from lda_generate_data import LdaDataGenerator
import numpy as np
import pylab as plt
import pandas as pd
import cPickle


class CollapseGibbsLda:
    
    def __init__(self, df, K, alpha, beta, random_state=None, previous_model=None, silent=False):
        """
        Initialises the collapsed Gibbs sampling for LDA
        
        Arguments:
        - df: the dataframe of counts of vocabularies x documents
        - K: no. of topics
        - alpha: symmetric prior on document-topic assignment
        - beta: symmetric prior on word-topic assignment
        - previous_model: previous LDA run, if any
        - silent: keep quiet and not print the progress
        """
        
        print "CGS LDA initialising"
        self.silent = silent
        self.df = df.replace(np.nan, 0)
        self.alpha = alpha
        self.beta = beta

        self.D = df.shape[0]    # total no of docs
        self.N = df.shape[1]    # total no of words

        # set total no of topics
        self.cv = False
        self.previous_model = previous_model
        if self.previous_model is not None:
            
            # if some old topics were fixed
            if hasattr(self.previous_model, 'selected_topics'):
            
                # then K is no. of old topics + no. of new topics
                self.K = K + self.previous_model.previous_K
                print "Total no. of topics = " + str(self.K)
                
                # extract the previous ckn and ck values for the old topics
                selected = self.previous_model.selected_topics
                self.previous_ckn = self.previous_model.ckn[selected, :]
                self.previous_ck = self.previous_model.ck[selected]
                
                # put them into the right shapes
                temp = np.zeros((self.K, self.N), int)
                self.previous_ckn = np.vstack((self.previous_ckn, temp)) # shape is (old_K+new_K) x N
                temp = np.zeros(self.K, int)
                self.previous_ck = np.hstack((self.previous_ck, temp)) # length is (old_K+new_K)
                
            else:
                
                # otherwise all previous topics were fixed, for cross-validation
                self.K = K
                self.previous_ckn = self.previous_model.ckn
                self.previous_ck = self.previous_model.ck
                self.cv = True
                
        else:

            self.K = K            
            self.previous_ckn = np.zeros((self.K, self.N), int)
            self.previous_ck = np.zeros(self.K, int)

        # make the current arrays too
        self.ckn = np.zeros((self.K, self.N), int)
        self.ck = np.zeros(self.K, int)
        self.cdk = np.zeros((self.D, self.K), int)
        self.cd = np.zeros(self.D, int)

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
            word_idx = self._word_indices(document)
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
            word_idx = self._word_indices(document)
            word_locs = []
            for pos, n in enumerate(word_idx):
                word_locs.append((pos, n))
            self.document_indices[d] = word_locs
                                                    
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
            from lda_cgs_numpy import sample_numpy
            sampler_func = sample_numpy
        else:
            print "Using JIT for LDA sampling"
            from lda_cgs_numba import sample_numba
            sampler_func = sample_numba

        # this will modify the various count matrices (Z, cdk, ckn, cd, ck) inside
        self.topic_word_, self.doc_topic_, self.all_lls = sampler_func(
                self.random_state, n_burn, n_samples, n_thin,
                self.D, self.N, self.K, self.document_indices,
                self.alpha, self.beta,
                self.Z, self.cdk, self.ckn, self.cd, self.ck,
                self.previous_ckn, self.previous_ck, self.silent, self.cv)
        self.loglikelihoods_ = self.all_lls
        
    def save(self, filename):
        # binary mode ('b') is required for portability between Unix and Windows
        f = file(filename, 'wb')
        cPickle.dump(self, f)
        f.close()
        print "Model saved to " + filename

    @classmethod
    def load(cls, filename):
        f = file(filename, 'rb')
        obj = cPickle.load(f)
        f.close()
        print "Model loaded from " + filename
        return obj
    
    def keep_topic(self, topic_indices):
        self.selected_topics = topic_indices
        self.previous_K = len(topic_indices)
                                    
    def _word_indices(self, document):
        """
        Turns a document vector of word counts into a vector of the indices
         words that have non-zero counts, repeated for each count
        e.g. 
        >>> word_indices(np.array([3, 0, 1, 2, 0, 5]))
        [0, 0, 0, 2, 3, 3, 5, 5, 5, 5, 5]
        """
        results = []
        for nnz in document.values.nonzero()[1]:
            for n in range(int(document[nnz])):
                results.append(nnz)
        return results

def main():

    multiplier = 1
    n_topics = 20 * multiplier
    n_docs = 100 * multiplier
    vocab_size = 200 * multiplier
    document_length = 50 * multiplier

    alpha = 0.1
    beta = 0.01    
    n_samples = 200
    n_burn = 100
    n_thin = 10
    
    vocab = []
    for n in range(vocab_size):
        vocab.append("word_" + str(n))

    gen = LdaDataGenerator(alpha, make_plot=True)
    # df = gen.generate_input_df(n_topics, vocab_size, document_length, n_docs, outfile='input/test1.csv')
    df = gen.generate_from_file('input/test1.csv')

    # for comparison
#     print "\nUsing LDA package"
#     gibbs = LDA(n_topics=n_topics, n_iter=n_samples, random_state=1, alpha=alpha, eta=beta)
#     start_time = time.time()
#     gibbs.fit(df.as_matrix())
#     print("--- TOTAL TIME %d seconds ---" % (time.time() - start_time))

#     print "\nUsing own LDA"
#     gibbs = CollapseGibbsLda(df, n_topics, alpha, beta, previous_model=None, silent=False)
#     start_time = time.time()
#     gibbs.run(n_burn, n_samples, n_thin, use_native=True)
#     print("--- TOTAL TIME %d seconds ---" % (time.time() - start_time))
     
    # try saving model
#     selected_topics = [0, 1, 2, 3, 4, 5]
#     gibbs.keep_topic(selected_topics)
#     gibbs.save('gibbs1.p')

    # try loading model
    gibbs = CollapseGibbsLda.load('gibbs1.p')
    print "Kept topics = " + str(gibbs.selected_topics)

    gen._plot_nicely(gibbs.doc_topic_.T, 'Inferred Topics X Docs', 'docs', 'topics', outfile='test1_doc_topic.png')
    gen._plot_nicely(gibbs.topic_word_, 'Inferred Topics X Terms', 'terms', 'topics', outfile='test1_topic_word.png')
    plt.plot(gibbs.all_lls)
    plt.show()
 
    topic_word = gibbs.topic_word_
    n_top_words = 20
    for i, topic_dist in enumerate(topic_word):
        topic_words = np.array(vocab)[np.argsort(topic_dist)][:-n_top_words:-1]
        print('Topic {}: {}'.format(i, ' '.join(topic_words)))
        
    # now run gibbs again on another df with the few selected topics above
    gen = LdaDataGenerator(alpha, make_plot=True)
    df2 = gen.generate_from_file('input/test2.csv')    
    gibbs2 = CollapseGibbsLda(df2, n_topics, alpha, beta, previous_model=gibbs, silent=False)
    gibbs2.run(n_burn, n_samples, n_thin, use_native=True)
 
    gen._plot_nicely(gibbs2.doc_topic_.T, 'Inferred Topics X Docs', 'docs', 'topics', outfile='test2_doc_topic.png')
    gen._plot_nicely(gibbs2.topic_word_, 'Inferred Topics X Terms', 'terms', 'topics', outfile='test2_topic_word.png')
    plt.plot(gibbs2.all_lls)
    plt.show()
  
    topic_word = gibbs2.topic_word_
    n_top_words = 20
    for i, topic_dist in enumerate(topic_word):
        topic_words = np.array(vocab)[np.argsort(topic_dist)][:-n_top_words:-1]
        print('Topic {}: {}'.format(i, ' '.join(topic_words)))

if __name__ == "__main__":
    main()