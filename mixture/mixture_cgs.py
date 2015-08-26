from collections import namedtuple
import sys
import time

import numpy as np
from numpy import int32
from numpy.random import RandomState
from scipy.misc import logsumexp

import justin.lda_utils as utils
from mixture_generate_data import MixtureDataGenerator

Sample = namedtuple('Sample', 'cdk ckn')

class CollapseGibbsMixture(object):

    def __init__(self, df, vocab, K, alpha, beta, random_state=None, previous_model=None):
        
        print "CGS Multinomial Mixture initialising"
        self.df = df.replace(np.nan, 0)
        self.alpha = alpha
        self.beta = beta
        self.D = df.shape[0]    # total no of docs
        self.N = df.shape[1]    # total no of words
        self.vocab = vocab
        assert(len(self.vocab)==self.N)

        self.cv = False
        self.previous_model = previous_model
        if self.previous_model is not None:
        
            # all previous topics were fixed, for cross-validation
            self.cv = True
            self.K = K
            self.previous_ckn = self.previous_model.ckn
            self.previous_ck = self.previous_model.ck
            self.previous_K = K

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
        self.cdk = np.zeros(self.K, int32)

        # make sure to get the same results from running gibbs each time
        if random_state is None:
            self.random_state = RandomState(1234567890)
        else:
            self.random_state = random_state

        # randomly assign documents to clusters
        self.Z = np.zeros(self.D, int32)
        for d in range(self.D):
            k = self.random_state.randint(self.K)
            if d%10==0:
                sys.stdout.write('.')
                sys.stdout.flush()
            document = self.df.iloc[[d]]
            word_idx = utils.word_indices(document)
            for pos, n in enumerate(word_idx):
                self.cdk[k] += 1
                self.ckn[k, n] += 1
                self.ck[k] += 1
                self.Z[d] = k
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
        
    def _get_posterior_probs(self, samp_cdk, samp_ckn):

        # update theta
        theta = samp_cdk + self.alpha 
        theta /= np.sum(theta)
        
        # update phi
        phi = samp_ckn + self.beta
        phi /= np.sum(phi, axis=1)[:, np.newaxis]
        
        return theta, phi
            
    def _get_perplexity(self, theta, phi):
        # for all documents and all terms
        marg = 0
        n_words = 0
        for d in range(self.D):
            document = self.df.iloc[[d]]
            nnz = document.values.nonzero()[1]
            doc_word_counts = document.values.flatten()
            nnz_counts = doc_word_counts[nnz]
            n_words += np.sum(nnz_counts)
            doc_marg = np.zeros(self.K)
            for k in range(self.K):
                temp = 0
                for n in nnz:
                    curr_word_count = doc_word_counts[n]
                    temp += curr_word_count * np.log(phi[k, n])
                doc_marg[k] = np.log(theta[k]) + temp
            marg += logsumexp(doc_marg)
        perp = np.exp(-(marg/n_words))
        return marg, perp        
            
    def _update_parameters(self):

        # use the last sample only
        if self.n_burn == 0:
            print "S=" + str(len(self.samples)) + ", using only the last sample."
            last_samp = self.samples[0]
            theta, phi = self._get_posterior_probs(last_samp.cdk, last_samp.ckn)            
            margs = []
            perps = []
            marg, perp = self._get_perplexity(theta, phi)
            margs.append(marg)
            perps.append(perp)
            return phi, theta, margs, perps

        print "S=" + str(len(self.samples)) + ", using all samples."
        thetas = []
        phis = []
        alphas = []
        for samp in self.samples:            
            theta, phi = self._get_posterior_probs(samp.cdk, samp.ckn)
            thetas.append(theta)
            phis.append(phi)
        
        # average over the results
        S = len(self.samples)
        
        print "Averaging over topic_words"
        avg_theta = np.zeros_like(thetas[0])
        for theta in thetas:
            avg_theta += theta
        avg_theta /= len(thetas)
        sys.stdout.flush()

        print "Averaging over doc_topics"
        avg_phi = np.zeros_like(phis[0])
        for phi in phis:
            avg_phi += phi
        avg_phi /= len(phis)
        sys.stdout.flush()

        print "Averaging over log evidence and perplexities"                
        margs = []
        perps = []
        for s in range(S):
            theta = thetas[s]
            phi = phis[s]
            marg, perp = self._get_perplexity(theta, phi)
            margs.append(marg)
            perps.append(perp)
        sys.stdout.flush()
        
        return avg_phi, avg_theta, margs, perps
                                                        
    def run(self, n_burn, n_samples, n_thin=1, use_native=True):
        
        self.n_burn = n_burn
        self.n_thin = n_thin
        if self.n_burn == 0:
            self.n_thin = 1
        
        """ 
        Runs the Gibbs sampling for multinomial mixture 
        
        Arguments:
        - n_burn: no of initial burn-in samples
        - n_samples: no of samples, must be > n_burn
        - n_thin: how often to thin the log_likelihood results stored
        - use_native: if True, will call the sampling function in lda_cgs_numba.py
        """

        # select the sampler function to use
        from mixture_cgs_numpy import sample_numpy
        sampler_func = None
        if not use_native:
            print "Using Numpy for mixture sampling"
            sampler_func = sample_numpy
        else:
            print "Using Numba for mixture sampling"
            try:
                from mixture_cgs_numba import sample_numba
                sampler_func = sample_numba
            except Exception, e:
                print "Error: %s" % e
                print "Numba not found. Using Numpy for mixture sampling"
                sampler_func = sample_numpy

        # this will modify the various count matrices (Z, cdk, ckn, cd, ck) inside
        self.loglikelihoods_, self.samples = sampler_func(
                self.random_state, n_burn, n_samples, n_thin,
                self.D, self.N, self.K, self.document_indices,
                self.alpha, self.beta,
                self.Z, self.cdk, self.previous_K,
                self.ckn, self.ck, self.previous_ckn, self.previous_ck)
        
        # update posterior alpha from the last sample  
        self.topic_word_, self.doc_topic_, self.margs, self.perps = self._update_parameters()

    def print_topic_words(self):
        topic_word = self.topic_word_
        n_top_words = 10
        for i, topic_dist in enumerate(topic_word):
            topic_words = np.array(self.vocab)[np.argsort(topic_dist)][:-n_top_words:-1]
            print('Cluster {}: {}'.format(i, ' '.join(topic_words)))                        

    @classmethod
    def load(cls, filename):
        raise NotImplementedError    

    def save(self, topic_indices, model_out, words_out):
        raise NotImplementedError    
        
    def visualise(self, topic_plotter):
        raise NotImplementedError    
        
def main():

    multiplier = 1
    n_cluster = 20 * multiplier
    n_docs = 100 * multiplier
    vocab_size = 200 * multiplier
    document_length = 50 * multiplier

    alpha = 0.1
    beta = 0.01    
    n_samples = 20
    n_burn = 10
    n_thin = 1

    random_state = RandomState(1234567890)

    gen = MixtureDataGenerator(alpha, make_plot=True)
#     df, vocab = gen.generate_input_df(n_cluster, vocab_size, document_length, n_docs, 
#                                       previous_vocab=None, vocab_prefix='gibbs1', 
#                                       df_outfile='input/test1.csv', vocab_outfile='input/test1.words')
    df, vocab = gen.generate_from_file('input/test1.csv', 'input/test1.words')

    mixture = CollapseGibbsMixture(df, vocab, n_cluster, alpha, beta, random_state=random_state)
    start_time = time.time()
    mixture.run(n_burn, n_samples, n_thin, use_native=True)
    print "margs"
    print mixture.margs
    print "perps"
    print mixture.perps
    print("--- TOTAL TIME %d seconds ---" % (time.time() - start_time))
    mixture.print_topic_words()
    
    gen._plot_nicely(mixture.topic_word_, 'Inferred Topics X Terms', 'terms', 'topics', outfile='test1_topic_word.png')

if __name__ == "__main__":
    main()