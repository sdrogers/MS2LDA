import sys

from lda_generate_data import LdaDataGenerator
import numpy as np
import pandas as pd
import pylab as plt


class CollapseGibbsLda:
    
    def __init__(self, df, K, alpha, beta):
        """
        Initialises the collaged Gibbs sampling for LDA
        
        Arguments:
        - df: the dataframe of counts of vocabularies x documents
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
            if d%10==0:
                sys.stdout.write('.')
                sys.stdout.flush()
            document = self.df.ix[:, d]
            word_idx = self._word_indices(document)
            for pos, n in enumerate(word_idx):
                k = np.random.randint(self.K)
                self.cdk[d, k] += 1
                self.cd[d] += 1
                self.ckn[k, n] += 1
                self.ck[k] += 1
                self.Z[(d, pos)] = k
        print
        
    def run(self, n_burn, n_samples):
        """ Runs the Gibbs sampling for LDA """
         
        for samp in range(n_samples):
        
            if samp >= n_burn:
                print "Sample "+ str(samp) + ' ',
            else:
                print "Burn-in " + str(samp) + ' ',
                
            for d in range(self.D):

                if d%10==0:
                    sys.stdout.write('.')
                    sys.stdout.flush()
                
                # turn word counts in the document into a vector of word occurences
                document = self.df.ix[:, d]
                word_idx = self._word_indices(document)

                for pos, n in enumerate(word_idx):
                    
                    # remove word from model
                    k = self.Z[(d, pos)]
                    self.cdk[d, k] -= 1
                    self.cd[d] -= 1
                    self.ckn[k, n] -= 1
                    self.ck[k] -= 1
 
                    # compute log prior and log likelihood
                    log_likelihood = self._log_likelihood(n)
                    log_prior = self._log_prior(d)
                    
                    # sample new k from the posterior distribution log_post
                    log_post = log_likelihood + log_prior
                    post = np.exp(log_post - log_post.max())
                    post = post / post.sum()
                    random_number = np.random.rand()
                    cumsum = np.cumsum(post)
                    k = 0
                    for k in range(len(cumsum)):
                        c = cumsum[k]
                        if random_number <= c:
                            break 
             
                    # reassign word back into model
                    self.cdk[d, k] += 1
                    self.cd[d] += 1
                    self.ckn[k, n] += 1
                    self.ck[k] += 1
                    self.Z[(d, pos)] = k

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


def main():

    alpha = 0.1
    beta = 0.1
    n_topics = 10
    n_docs = 20
    vocab_size = 100
    document_length = 50

    gen = LdaDataGenerator(alpha)
    df = gen.generate_input_df(n_topics, vocab_size, document_length, n_docs)
    
    gibbs = CollapseGibbsLda(df, n_topics, alpha, beta)
    gibbs.run(n_burn=100, n_samples=200)
    
    gen._plot_nicely(gibbs.phi, 'Inferred Topics X Vocabularies', 'vocabs', 'topics', 'inferred_topic_vocab.png')
    gen._plot_nicely(gibbs.theta.T, 'Inferred Topics X Docs', 'docs', 'topics', 'inferred_topic_docs.png')

if __name__ == "__main__":
    main()