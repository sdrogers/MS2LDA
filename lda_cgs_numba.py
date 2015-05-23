import math
import sys

from numba import jit
from numba.types import int64, float64

import numpy as np

def sample_numba(random_state, n_burn, n_samples, n_thin, 
            D, N, K, document_indices, 
            alpha, beta, 
            Z, cdk, ckn, cd, ck,
            is_training, previous_model, silent):

    all_lls = []
    thin = 0
    
    # prepare some K-length vectors to hold the intermediate results during loop
    post = np.empty(K, dtype=np.float64)
    cumsum = np.empty(K, dtype=np.float64)

    # loop over samples
    for samp in range(n_samples):
    
        s = samp+1        
        if not silent:
            if s >= n_burn:
                print("Sample " + str(s) + " "),
            else:
                print("Burn-in " + str(s) + " "),

        # loop over documents
        for d in range(D):

            if not silent and d%10==0:                        
                sys.stdout.write('.')
                sys.stdout.flush()

            # loop over words, not so easy to JIT due to rng and Z
            word_locs = document_indices[d]
            for pos, n in word_locs:
                random_number = random_state.rand()                
                k = Z[(d, pos)]                
                k = _nb_get_new_index(d, n, k, cdk, cd, ck, ckn,
                                      N, K, alpha, beta, 
                                      post, cumsum, random_number)
                Z[(d, pos)] = k

        if s > n_burn:
            thin += 1
            if thin%n_thin==0:    
                ll = _nb_ll(D, N, K, alpha, beta, ckn, ck)
                all_lls.append(ll)      
                if not silent: print(" Log likelihood = %.3f " % ll)                        
            else:                
                if not silent: print
        else:
            if not silent: print
            
    # update phi
    phi = ckn + beta
    phi /= np.sum(phi, axis=1)[:, np.newaxis]

    # update theta
    theta = cdk + alpha 
    theta /= np.sum(theta, axis=1)[:, np.newaxis]

    all_lls = np.array(all_lls)            
    return phi, theta, all_lls

@jit(int64(
           int64, int64, int64, int64[:, :], int64[:], int64[:], int64[:, :],
           int64, int64, float64, float64,
           float64[:], float64[:], float64
), nopython=True)
def _nb_get_new_index(d, n, k, cdk, cd, ck, ckn,
                      N, K, alpha, beta, 
                      post, cumsum, random_number):

    temp_ckn = ckn[:, n]
    temp_cdk = cdk[d, :]

    # remove from model
    cdk[d, k] -= 1
    cd[d] -= 1                    
    ck[k] -= 1
    ckn[k, n] -= 1

    # log_likelihood = np.log(ckn[:, n] + beta) - np.log(ck + N*beta)
    # log_prior = np.log(cdk[d, :] + alpha) - np.log(cd[d] + K*alpha)        
    # log_post = log_likelihood + log_prior
    
    # we risk underflowing by not working in log space here
    for i in range(len(post)):
        likelihood = (temp_ckn[i] + beta) / (ck[i] + N*beta)
        prior = (temp_cdk[i] + alpha) / (cd[d] + K*alpha)
        post[i] = likelihood * prior

    # post = np.exp(log_post - log_post.max())
    # post = post / post.sum()
    sum_post = 0
    for i in range(len(post)):
        sum_post += post[i]
    for i in range(len(post)):
        post[i] = post[i] / sum_post
                            
    # k = np.random.multinomial(1, post).argmax()
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
    cdk[d, k] += 1
    cd[d] += 1
    ck[k] += 1
    ckn[k, n] += 1
    
    return k

@jit(float64(int64, int64, int64, float64, float64, int64[:, :], int64[:]), nopython=True)
def _nb_ll(D, N, K, alpha, beta, ckn, ck):
    ll = K * ( math.lgamma(N*beta) - (math.lgamma(beta)*N) )
    for k in range(K):
        for n in range(N):
            ll += math.lgamma(ckn[k, n]+beta)
        ll -= math.lgamma(ck[k] + N*beta)
    ll += D * ( math.lgamma(K*alpha) - (math.lgamma(alpha)*K) )
    return ll