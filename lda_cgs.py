"""
Implementation of collapsed Gibbs sampling for LDA
inspired by https://gist.github.com/mblondel/542786
"""

import numpy as np
import pandas as pd
import pylab as plt

class CollapseGibbsLda:
    
    def __init__(self, df, K, alpha, beta):
        """
        Initialises the collaged Gibbs sampling for LDA
        
        Arguments:
        - df: the dataframe of counts of words x docs
        - K: no. of topics
        - alpha: symmetric prior on document-topic assignment
        - beta: symmetric prior on word-topic assignment
        """
        
        self.df = df.replace(np.nan, 0)
        self.alpha = alpha
        self.beta = beta

        self.K = K              # total no of topics
        self.N = df.shape[0]    # total no of words
        self.D = df.shape[1]    # total no of docs
        self.Z = {}
        self.cdk = np.zeros((self.D, self.K))
        self.ckn = np.zeros((self.K, self.N))
        self.cd = np.zeros(self.D)
        self.ck = np.zeros(self.K)
        
        # randomly assign words to topics
        for d in range(self.D):
            document = self.df.ix[:, d]
            word_idx = self._word_indices(document)
            for n in word_idx:
                k = np.random.randint(self.K)
                self.cdk[d, k] += 1
                self.cd[d] += 1
                self.ckn[k, n] += 1
                self.ck[k] += 1
                self.Z[(d, n)] = k
        
    def run(self, n_burn, n_samples):
        """ Runs the Gibbs sampling for LDA """
         
        for samp in range(n_samples):
        
            if samp > n_burn:
                print "Burn-in "+ str(samp)
            else:
                print "Sample " + str(samp)
                
            for d in range(self.D):
                
                # turn word counts in the document into a vector of word occurences
                document = self.df.ix[:, d]
                word_idx = self._word_indices(document)

                for n in word_idx:
                    
                    # remove word from model
                    k = self.Z[(d, n)]
                    self.cdk[d, k] -= 1
                    self.cd[d] -= 1
                    self.ckn[k, n] -= 1
                    self.ck[k] -= 1
 
                    # sample new topic
                    p_z = self._conditional_distribution(d, n)
                    k = self._sample_index(p_z)
 
                    # reassign word back into model
                    self.cdk[d, k] -= 1
                    self.cd[d] -= 1
                    self.ckn[k, n] -= 1
                    self.ck[k] -= 1
                    self.Z[(d, n)] = k
                    
    def _word_indices(document):
        """
        Turns a document vector of word counts into a vector of the indices
         words that have non-zero counts, repeated for each count
        e.g. 
        >>> word_indices(np.array([3, 0, 1, 2, 0, 5]))
        [0, 0, 0, 2, 3, 3, 5, 5, 5, 5, 5]
        """
        results = []
        for nnz in document.nonzero()[0]:
            for n in range(int(document[nnz])):
                results.append(nnz)
        return results

    def _conditional_distribution(self, d, n):
        print "TODO"

    def _sample_index(self, p_z):
        print "TODO"
        


def main():

    df = pd.read_csv('input/Beer_3_T10_POS_fragments.csv', index_col=0)
    vocabs = df.index.values
    print vocabs
    
    gibbs = CollapseGibbsLda(df, 10, 0.1, 0.1)
    gibbs.run(0, 100)

if __name__ == "__main__":
    main()