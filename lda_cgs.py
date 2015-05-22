"""
Implementation of collapsed Gibbs sampling for LDA
"""

from math import log
from scipy import stats
from scipy import weave
from scipy.special import gammaln
from scipy.weave import blitz
from scipy.weave import inline
import sys

from lda_generate_data import LdaDataGenerator
import numpy as np
import pandas as pd
import pylab as plt
import time

class CollapseGibbsLda:
    
    def __init__(self, df, K, alpha, beta, previous_model=None, use_inline=True):
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
        self.D = df.shape[0]    # total no of docs
        self.N = df.shape[1]    # total no of words
        self.Z = {}

        if previous_model is None:
            self.is_training = True
        else:
            self.is_training = False
            self.previous_model = previous_model

        self.cdk = np.zeros((self.D, self.K), int)
        self.cd = np.zeros(self.D, int)
        self.ckn = np.zeros((self.K, self.N), int)
        self.ck = np.zeros(self.K, int)

        # randomly assign words to topics if training
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
            
        self.use_inline = use_inline
        
    def run(self, n_burn, n_samples, n_thin):
        """ Runs the Gibbs sampling for LDA """
                
        self.all_lls = []
        thin = 0
        for samp in range(n_samples):
        
            s = samp+1        
            if s >= n_burn:
                print "Sample "+ str(s) + ' ',
            else:
                print "Burn-in " + str(s) + ' ',
                
            for d in range(self.D):

                if d%10==0:
                    
                    sys.stdout.write('.')
                    sys.stdout.flush()
                
                word_idx = self.document_indices[d]
                for pos, n in enumerate(word_idx):
                    
                    # remove word from model
                    k = self.Z[(d, pos)]
                    self.cdk[d, k] -= 1
                    self.cd[d] -= 1                    
                    self.ck[k] -= 1
                    self.ckn[k, n] -= 1
 
                    # compute log prior and log likelihood
                    log_likelihood = self._compute_left(n)
                    log_prior = self._compute_right(d)
                    
                    # sample new k from the posterior distribution log_post
                    log_post = log_likelihood + log_prior
                    post = np.exp(log_post - log_post.max())
                    post = post / post.sum()
#                     random_number = np.random.rand()
#                     cumsum = np.cumsum(post)
#                     k = 0
#                     for k in range(len(cumsum)):
#                         c = cumsum[k]
#                         if random_number <= c:
#                             break 
                    k = np.random.multinomial(1, post).argmax()
             
                    # reassign word back into model
                    self.cdk[d, k] += 1
                    self.cd[d] += 1
                    self.ck[k] += 1
                    self.ckn[k, n] += 1
                    self.Z[(d, pos)] = k

            if s > n_burn:
                thin += 1
                if thin%n_thin==0:    
                    ll = self._log_likelihood()
                    self.all_lls.append(ll)      
                    print " Log likelihood = %.3f" % ll
                else:                
                    print
            else:
                print
                
        # update phi
        self.phi = self.ckn + self.beta
        self.phi /= np.sum(self.phi, axis=1)[:, np.newaxis]
        self.topic_word_ = self.phi

        # update theta
        self.theta = self.cdk + self.alpha 
        self.theta /= np.sum(self.theta, axis=1)[:, np.newaxis]
        self.doc_topic_ = self.theta                

        self.all_lls = np.array(self.all_lls)
                                    
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

    def _compute_left(self, n):
        """ Computes p(w|z, ...) """
        
        if self.use_inline:
            
            # C version
            ckn = self.ckn[:, n]
            ck = self.ck
            assert len(ckn) == len(ck)
            arr_len = len(ckn)
            beta = self.beta
            N = self.N
            results = np.zeros((1, arr_len))
            code = """
                for (int i=0; i<arr_len; i++)
                {
                    long int ckn_i = ckn(i);
                    double temp1 = log(ckn_i + beta);
                    long int ck_i = ck(i);
                    double temp2 = log(ck_i + N*beta);
                    results(0, i) = temp1 - temp2;
                    // printf("%.7f ", results(0, i));
                }
                // printf("\\n");
            """
            inline(code, ['ckn', 'ck', 'arr_len', 'beta', 'N', 'results'], 
                                    type_converters = weave.converters.blitz, 
                                    compiler = "gcc", headers=["<math.h>"])
            log_likelihood = results[0] # since results is 1 x arr_len

        else:
            
            # slow python version
            if self.is_training:
                log_likelihood = np.log(self.ckn[:, n] + self.beta) - np.log(self.ck + self.N*self.beta)
            else:
                log_likelihood = np.log(self.ckn[:, n] + self.previous_model.ckn[:, n] + self.beta) - \
                    np.log(self.ck + self.previous_model.ckn[:, n] + self.N*self.beta)            

        return log_likelihood
            
    def _compute_right(self, d):
        """ Computes p(z) """

        if self.use_inline:
            
            # C version
            cdk = self.cdk[d, :]
            cd = int(self.cd[d])
            arr_len = len(cdk)
            alpha = self.alpha
            K = self.K
            results = np.zeros((1, arr_len))
            code = """
                for (int i=0; i<arr_len; i++)
                {
                    long int cdk_i = cdk(i);
                    double temp1 = log(cdk_i + alpha);
                    double temp2 = log(cd + K*alpha);
                    results(0, i) = temp1 - temp2;
                    // printf("%.7f ", results(0, i));
                }
                // printf("\\n");
            """
            inline(code, ['cdk', 'cd', 'arr_len', 'alpha', 'K', 'results'], 
                                    type_converters = weave.converters.blitz, 
                                    compiler = "gcc", headers=["<math.h>"])
            log_prior = results[0] # since results is 1 x arr_len

        else:

            # slow python version                
            log_prior = np.log(self.cdk[d, :] + self.alpha) - np.log(self.cd[d] + self.K*self.alpha)
        
        return log_prior
    
    def _log_likelihood(self):
        """ Computes the log likelihood of the data """
        
        ll = self.K * ( gammaln(self.N*self.beta) - (gammaln(self.beta)*self.N) )
        for k in range(self.K):
            for n in range(self.N):
                ll += gammaln(self.ckn[k, n]+self.beta)
            ll -= gammaln(self.ck[k] + self.N*self.beta)
           
#         ll += self.D * ( gammaln(self.K*self.alpha) - (gammaln(self.alpha)*self.K) )
#         for d in range(self.D):
#             for k in range(self.K):
#                 ll += gammaln(self.cdk[d, k]+self.alpha)
#             ll -= gammaln(self.cd[d] + self.K*self.alpha)
            
        return ll

def main():

    n_topics = 10
    alpha = 0.1
    beta = 0.01    
    n_docs = 50
    vocab_size = 200
    document_length = 50
    gen = LdaDataGenerator(alpha)
    df = gen.generate_input_df(n_topics, vocab_size, document_length, n_docs)

    start_time = time.time()
    gibbs = CollapseGibbsLda(df, 13, alpha, beta, use_inline=True)
    gibbs.run(n_burn=100, n_samples=200, n_thin=1)
    print("--- TOTAL TIME %d seconds ---" % (time.time() - start_time))
        
    gen._plot_nicely(gibbs.phi, 'Inferred Topics X Terms', 'terms', 'topics')
    gen._plot_nicely(gibbs.theta.T, 'Inferred Topics X Docs', 'docs', 'topics')
    plt.plot(gibbs.all_lls)
    plt.show()

if __name__ == "__main__":
    main()