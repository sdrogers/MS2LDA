import math
import sys

from numba import jit
from numba.types import int32, float64

from mixture_cgs import Sample
import numpy as np

def sample_numba(random_state, n_burn, n_samples, n_thin, 
            D, N, K, document_indices, 
            alpha, beta, 
            Z, cdk, previous_K,
            ckn, ck, previous_ckn, previous_ck):
    
    # prepare some K-length vectors to hold the intermediate results during loop
    post = np.empty(K, dtype=np.float64)
    cumsum = np.empty(K, dtype=np.float64)

    # precompute repeated constants
    N_beta = np.sum(beta)
    K_alpha = np.sum(alpha)    
    
    # loop over samples
    samples = []
    all_lls = []
    thin = 0
    for samp in range(n_samples):
    
        s = samp+1        
        if s > n_burn:
            print("Sample " + str(s) + " "),
        else:
            print("Burn-in " + str(s) + " "),

        # loop over documents
        for d in range(D):
            
            random_number = random_state.rand()
            word_idx = np.array(document_indices[d])
            word_idx = word_idx[:, 1]
            
            # assign new k
            k = Z[d]
            k = _nb_get_new_index(d, k, cdk, word_idx,
                                  N, K, previous_K, alpha, beta, 
                                  N_beta, K_alpha,
                                  post, cumsum, random_number,
                                  ckn, ck, previous_ckn, previous_ck)
            Z[d] = k
    
        ll = _nb_ll(D, N, K, alpha, beta, N_beta, K_alpha, cdk, ckn, ck)        
        all_lls.append(ll)      
        print(" Log likelihood = %.3f " % ll)
            
        # store all the samples after thinning
        if n_burn > 0 and s > n_burn:
            thin += 1
            if thin%n_thin==0:
                cdk_copy = np.copy(cdk)
                ckn_copy = np.copy(ckn)
                to_store = Sample(cdk_copy, ckn_copy)
                samples.append(to_store)                                      
                
    # store the last sample only
    if n_burn == 0:
        cdk_copy = np.copy(cdk)
        ckn_copy = np.copy(ckn)
        to_store = Sample(cdk_copy, ckn_copy)
        samples.append(to_store)                                      
            
    all_lls = np.array(all_lls)            
    return all_lls, samples

def _nb_get_new_index(d, k, cdk, word_idx,
                      N, K, previous_K, alpha, beta, 
                      N_beta, K_alpha,                      
                      post, cumsum, random_number, 
                      ckn, ck, previous_ckn, previous_ck):

    # remove from model
    cdk[k] -= 1
    for n in word_idx:
        ckn[k, n] -= 1
        ck[k] -= 1
    
    # compute likelihood, prior, posterior
    for i in range(len(post)):
        likelihood = 0
        prior = math.log(cdk[i] + alpha[i])
        for n in word_idx:
            if i < previous_K:
                likelihood += math.log(previous_ckn[i, n] + beta[n]) - math.log(previous_ck[i] + N_beta)
            else:
                likelihood += math.log(ckn[i, n] + beta[n]) - math.log(ck[i] + N_beta)
        post[i] = likelihood + prior
        
    # sample new k from the posterior distribution log_post
    max_log_post = post[0]
    for i in range(len(post)):
        val = post[i]
        if val > max_log_post:
            max_log_post = val
    sum_post = 0
    for i in range(len(post)):
        post[i] = math.exp(post[i] - max_log_post)
        sum_post += post[i]
    for i in range(len(post)):
        post[i] = post[i] / sum_post

    total = 0
    for i in range(len(post)):
        val = post[i]
        total += val
        cumsum[i] = total
    k = 0
    for k in range(len(cumsum)):
        c = cumsum[k]
        if random_number <= c:
            break 

    # put back to model
    cdk[k] += 1
    for n in word_idx:    
        ckn[k, n] += 1
        ck[k] += 1
    
    return k

@jit(float64(int32, int32, int32, float64[:], float64[:], float64, float64, int32[:], int32[:, :], int32[:]), nopython=True)
def _nb_ll(D, N, K, alpha, beta, N_beta, K_alpha, cdk, ckn, ck):
    
    temp_sum = 0
    for b in beta:
        temp_sum += math.lgamma(b)
    ll = K * ( math.lgamma(N_beta) - temp_sum )
    for k in range(K):
        for n in range(N):
            ll += math.lgamma(ckn[k, n]+beta[n])
        ll -= math.lgamma(ck[k] + N_beta)

    temp_sum = 0
    for a in alpha:
        temp_sum += math.lgamma(a)
    ll += math.lgamma(K_alpha) - temp_sum
    cdk_sum = 0
    for k in range(K):
        ll += math.lgamma(cdk[k]+alpha[k])
        cdk_sum += cdk[k]
    ll -= math.lgamma(cdk_sum + K_alpha)                
    
    return ll