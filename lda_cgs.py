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
        
        print "CGS LDA initialising ",
        self.df = df.replace(np.nan, 0)
        self.alpha = alpha
        self.beta = beta

        self.K = K              # total no of topics
        self.D = df.shape[0]    # total no of docs
        self.N = df.shape[1]    # total no of words
        self.Z = {}        
        if random_state is None:
            self.random_state = RandomState(1234567890)
        else:
            self.random_state = random_state

        self.cdk = np.zeros((self.D, self.K), int)
        self.cd = np.zeros(self.D, int)
        self.ckn = np.zeros((self.K, self.N), int)
        self.ck = np.zeros(self.K, int)

        # randomly assign words to topics
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
                self.ck[k] += 1
                self.ckn[k, n] += 1
                self.Z[(d, pos)] = k
        print

        self.document_indices = {}
        for d in range(self.D):
            # turn word counts in the document into a vector of word occurences
            document = self.df.iloc[[d]]
            word_idx = self._word_indices(document)
            word_locs = []
            for pos, n in enumerate(word_idx):
                word_locs.append((pos, n))
            self.document_indices[d] = word_locs

        if previous_model is None:
            self.is_training = True
            self.previous_model = None
        else:
            self.is_training = False
            self.previous_model = previous_model
        self.silent = silent
                                        
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
                self.random_state,
                n_burn, n_samples, n_thin,
                self.D, self.N, self.K, self.document_indices,
                self.alpha, self.beta,
                self.Z, self.cdk, self.ckn, self.cd, self.ck,
                self.is_training, self.previous_model, self.silent)
                                    
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

    multiplier = 10
    n_topics = 20 * multiplier
    n_docs = 100 * multiplier
    vocab_size = 200 * multiplier
    document_length = 50 * multiplier

    alpha = 0.1
    beta = 0.01    
    n_samples = 100
    n_burn = 50
    n_thin = 1

    gen = LdaDataGenerator(alpha, make_plot=False)
    df = gen.generate_input_df(n_topics, vocab_size, document_length, n_docs)

    # for comparison
#     print "\nUsing LDA package"
#     gibbs = LDA(n_topics=n_topics, n_iter=n_samples, random_state=1, alpha=alpha, eta=beta)
#     start_time = time.time()
#     gibbs.fit(df.as_matrix())
#     print("--- TOTAL TIME %d seconds ---" % (time.time() - start_time))

    print "\nUsing own LDA"
    gibbs = CollapseGibbsLda(df, n_topics, alpha, beta, previous_model=None, silent=False)
    start_time = time.time()
    gibbs.run(n_burn, n_samples, n_thin, use_native=False)
    print("--- TOTAL TIME %d seconds ---" % (time.time() - start_time))
        
    gen._plot_nicely(gibbs.doc_topic_.T, 'Inferred Topics X Docs', 'docs', 'topics')
    gen._plot_nicely(gibbs.topic_word_, 'Inferred Topics X Terms', 'terms', 'topics')
    plt.plot(gibbs.all_lls)
    plt.show()

if __name__ == "__main__":
    main()