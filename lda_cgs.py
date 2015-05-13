"""
Implementation of collapsed Gibbs sampling for LDA
inspired by https://gist.github.com/mblondel/542786, based on

Griffiths, Thomas L., and Mark Steyvers. "Finding scientific topics." 
Proceedings of the National Academy of Sciences 101.suppl 1 (2004): 5228-5235.
"""

import numpy as np
import pandas as pd
import pylab as plt

import sys

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
        
        print "CGS LDA initialising ",
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
            sys.stdout.write('.')
            sys.stdout.flush()
            document = self.df.ix[:, d]
            word_idx = self._word_indices(document)
            for n in word_idx:
                k = np.random.randint(self.K)
                self.cdk[d, k] += 1
                self.cd[d] += 1
                self.ckn[k, n] += 1
                self.ck[k] += 1
                self.Z[(d, n)] = k
        print
        
    def run(self, n_burn, n_samples):
        """ Runs the Gibbs sampling for LDA """
         
        for samp in range(n_samples):
        
            if samp >= n_burn:
                print "Sample "+ str(samp) + ' ',
            else:
                print "Burn-in " + str(samp) + ' ',
                
            for d in range(self.D):

                sys.stdout.write('.')
                sys.stdout.flush()
                
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
 
                    # compute log prior and likelihood
                    log_likelihood = self._log_likelihood(n)
                    log_prior = self._log_prior(d)
                    
                    # sample new k from the posterior distribution
                    log_post = log_likelihood + log_prior
                    k = self._sample_posterior(log_post)
 
                    # reassign word back into model
                    self.cdk[d, k] -= 1
                    self.cd[d] -= 1
                    self.ckn[k, n] -= 1
                    self.ck[k] -= 1
                    self.Z[(d, n)] = k

            if samp > n_burn:                    
                # update phi
                self.phi = self.ckn + self.beta
                self.phi /= np.sum(self.phi, axis=1)[:, np.newaxis]
                # update theta
                self.theta = self.cdk + self.alpha 
                self.theta /= np.sum(self.theta, axis=1)[:, np.newaxis]
                
            print
                    
    def _word_indices(self, document):
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

    def _log_likelihood(self, n):
        """ Computes likelihood p(w|z, ...) """
        log_likelihood = np.log(self.ckn[:, n] + self.beta) - np.log(self.ck + self.N*self.beta)
        return log_likelihood
    
    def _log_prior(self, d):
        """ Computes prior p(z) """
        log_prior = np.log(self.cdk[d,:] + self.alpha) - np.log(self.cd[d] + self.K*self.alpha)
        return log_prior

    def _sample_posterior(self, log_post):
        """ Samples from the posterior distribution p(z|...) """
        log_post = np.exp(log_post - log_post.max())
        log_post = log_post / log_post.sum()
        random_number = np.random.rand()
        cumsum = np.cumsum(log_post)
        k = 0
        for k in range(len(cumsum)):
            c = cumsum[k]
            if random_number <= c:
                break        
        return k


def main():

    df = pd.read_csv('input/Beer_3_T10_POS_fragments.csv', index_col=0)

    # discretise, log and scale df from 0 .. 100
    df = np.log10(df)
    df /= df.max().max()
    df *= 100
            
    # get rid of NaNs and floor the values
    df = df.replace(np.nan,0)
    df = df.apply(np.floor)    
    print "Data shape " + str(df.shape)
    
    gibbs = CollapseGibbsLda(df, K=10, alpha=0.1, beta=0.1)
    gibbs.run(n_burn=10, n_samples=20)
    print "phi = " + str(gibbs.phi)
    print "theta = " + str(gibbs.theta)    

if __name__ == "__main__":
    main()