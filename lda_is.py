import timeit

from numba.decorators import jit
from numba.types import int32, float64
from numpy.random import rand
from scipy.special import gammaln
from scipy.misc import logsumexp

import numpy as np
import pylab as plt
import scipy.io as sio


@jit(int32[:](float64[:], float64[:]), nopython=True)
def sample_cumsum(random_numbers, cs):
    num_samples = len(random_numbers)
    sampled_idx = np.empty(num_samples, int32)
    for r in range(num_samples):
        random_number = random_numbers[r]
        k = 0
        for k in range(len(cs)):
            c = cs[k]
            if random_number <= c:
                break 
        sampled_idx[r] = k
    return sampled_idx

@jit(int32[:, :](int32, int32[:, :], int32), nopython=True)
def count_member(num_samples, samples, T):
    
    Nk = np.zeros((T, num_samples), dtype=np.int32) # T x num_samples    
    for s in range(num_samples):
        
        # temp = np.bincount(samples[:, s], minlength=T)
        # Nk[:, s] = temp

        # faster bincount
        temp = np.zeros(T, dtype=np.int32)
        for k in samples[:, s]:
            temp[k] += 1
        for k in range(T):
            Nk[k, s] = temp[k]
    
    return Nk

# this is a Numpy port of the Matlab's importance sampling method to approximate LDA marginal likelihood,
# as described in Wallach, et al. (2009). "Evaluation methods for topic models."
def ldae_is_variants(words, topics, topic_prior, num_samples=1000, variant=3, variant_iters=1000):

    (T, V) = topics.shape
    Nd = len(words)
    topic_alpha = np.sum(topic_prior)

    # Creating the proposal q-distribution
    if variant == 1:
        # Importance sample from prior
        qstar = np.tile(topic_prior, (1, Nd)) # T x Nd
        qq = qstar / np.sum(qstar, 0)
    else:
        # Take w_n into account when picking z_n
        topic_word_dist = topics[:, words]
        qstar = np.multiply(topic_prior, topic_word_dist) # T x Nd
        qq = qstar / np.sum(qstar, 0)

        if variant == 3:
            for i in range(variant_iters):
                # Now create pseudo-counts from qq and recompute qq using them
                temp = topic_prior.flatten() + np.sum(qq, 1)
                temp = temp[:, None]
                pseudo_counts = temp - qq

                topic_word_dist = topics[:, words]
                qstar = np.multiply(pseudo_counts, topic_word_dist) # T x Nd
                qq = qstar / np.sum(qstar, 0)

    # Drawing samples from the q-distribution
    samples = np.zeros((Nd, num_samples), dtype=np.int32)
    for n in range(Nd):
        probs = qq[:, n]
        cs = np.cumsum(probs)
        random_numbers = np.random.random(num_samples)
        sampled_idx = sample_cumsum(random_numbers, cs)
        # sampled_idx = np.digitize(random_numbers, cs)
        samples[n, :] = sampled_idx

    # Do a bin count for each topic within a sample
    Nk = count_member(num_samples, samples, T)

    # Evaluate P(z, v) at samples and compare to q-distribution
    log_pz = np.sum(gammaln(Nk+topic_prior), 0) + \
             gammaln(topic_alpha) - np.sum(gammaln(topic_prior)) \
             - gammaln(Nd+topic_alpha)      # length is num_samples
    log_w_given_z = np.zeros(num_samples)   # length is num_samples
    for n in range(Nd):
        sampled_topic_idx = samples[n, :]
        word_idx = words[n]
        topic_word_prob = topics[sampled_topic_idx, word_idx]
        log_w_given_z = log_w_given_z + np.log(topic_word_prob)
    log_joint = log_pz + log_w_given_z      # length is num_samples

    log_qq = np.zeros(num_samples)          # length is num_samples
    for n in range(Nd):
        sampled_topic_idx = samples[n, :]
        sampled_qq = qq[sampled_topic_idx, n]
        log_qq = log_qq + np.log(sampled_qq)

    log_weights = log_joint - log_qq

    # the logsumexp below might underflow .. !!
    # log_evidence = np.log(np.sum(np.exp(log_weights))) - np.log(len(log_weights))
    log_evidence = logsumexp(log_weights) - np.log(len(log_weights))
    return log_evidence

def generate_synthetic():

    T = 3;
    V = 5;
    Nd = 7;

    # make some topics distributions
    topics = rand(T, V)
    rowsum = np.sum(topics, 1)
    for i in range(T):
        topics[i, :] = topics[i, :] / rowsum[i] # normalise each row to sum to 1

    # make the prior for the topics
    topic_prior = rand(T, 1)
    topic_prior = topic_prior / np.sum(topic_prior)

    # make the words
    words = np.floor(rand(Nd) * V)
    words = words.astype(np.int32)

    return words, topics, topic_prior

def generate_from_matlab(matfile):    
    mat_contents = sio.loadmat(matfile)
    topic_prior = mat_contents['topic_prior']
    words = mat_contents['words']
    words = words.astype(np.int32)
    words = words-1 # because matlab indexes from 1 ..
    words = words.flatten()
    topics = mat_contents['topics']
    return words, topics, topic_prior

def main():

    num_samples = 1000000
    variant = 3
    variant_iters = 1000

    words, topics, topic_prior = generate_synthetic()
    # words, topics, topic_prior = generate_from_matlab('/home/joewandy/sampling codes/lda_eval_matlab_code_20120930/test.mat')

    results = []
    for x in range(30):
        start = timeit.default_timer()
        is_pseudopost = ldae_is_variants(words, topics, topic_prior, num_samples, variant, variant_iters)
        print "is_pseudopost = " + str(is_pseudopost)
        stop = timeit.default_timer()
        print "DONE. Time=" + str(stop-start)        
        results.append(is_pseudopost)

    results = np.array(results)
    print results
    print np.mean(results)
    plt.boxplot(results)
    plt.show()

if __name__ == "__main__":
    main()
