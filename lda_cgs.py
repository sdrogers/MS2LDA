"""
Implementation of collapsed Gibbs sampling for LDA
"""

import sys
import time

from lda_generate_data import LdaDataGenerator
import numpy as np


class CollapseGibbsLda:
    
    def __init__(self, df, K, alpha, beta, previous_model=None):
        """
        Initialises the collaged Gibbs sampling for LDA
        
        Arguments:
        - df: the dataframe of counts of vocabularies x documents
        - K: no. of topics
        - alpha: symmetric prior on document-topic assignment
        - beta: symmetric prior on word-topic assignment
        - previous_model: previous LDA run, if any
        """
        
        print "CGS LDA initialising ",
        self.df = df.replace(np.nan, 0)
        self.alpha = alpha
        self.beta = beta

        self.K = K              # total no of topics
        self.D = df.shape[0]    # total no of docs
        self.N = df.shape[1]    # total no of words
        self.Z = {}

        if previous_model is None:
            self.is_training = True
            self.previous_model = None
        else:
            self.is_training = False
            self.previous_model = previous_model

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
                k = np.random.randint(self.K)
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
            self.document_indices[d] = word_idx
                                        
    def run(self, n_burn, n_samples, n_thin, use_native=False):
        """ Runs the Gibbs sampling for LDA """

        # select the sampler function to use
        sampler_func = None
        if not use_native:
            print "Using Numpy sampling"
            from lda_cgs_numpy import sample_numpy
            sampler_func = sample_numpy
        else:
            print "Using compiled sampling"
            from lda_cgs_numba import sample_numba
            sampler_func = sample_numba

        # this will modify the various count matrices (Z, cdk, ckn, cd, ck) inside
        self.topic_word_, self.doc_topic_, self.all_lls = sampler_func(
                n_burn, n_samples, n_thin,
                self.D, self.N, self.K, self.document_indices,
                self.alpha, self.beta,
                self.Z, self.cdk, self.ckn, self.cd, self.ck,
                self.is_training, self.previous_model)
                                    
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

    n_topics = 200
    alpha = 0.1
    beta = 0.01    
    n_docs = 1000
    vocab_size = 2000
    document_length = 600
    gen = LdaDataGenerator(alpha)
    df = gen.generate_input_df(n_topics, vocab_size, document_length, n_docs)

    start_time = time.time()
    gibbs = CollapseGibbsLda(df, n_topics, alpha, beta, previous_model=None)
    gibbs.run(n_burn=0, n_samples=1, n_thin=1, use_native=False)
    print("--- TOTAL TIME %d seconds ---" % (time.time() - start_time))
        
#     gen._plot_nicely(gibbs.topic_word_, 'Inferred Topics X Terms', 'terms', 'topics')
#     gen._plot_nicely(gibbs.doc_topic_.T, 'Inferred Topics X Docs', 'docs', 'topics')
#     plt.plot(gibbs.all_lls)
#     plt.show()

if __name__ == "__main__":
    main()