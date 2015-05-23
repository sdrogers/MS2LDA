import math
import sys

from numba import jit
from numba.decorators import autojit
from numba.npyufunc.decorators import vectorize
from numba.types import double, int64
from scipy.special import gammaln

import numpy as np


def sample_numba(n_burn, n_samples, n_thin, 
            D, N, K, document_indices, 
            alpha, beta, 
            Z, cdk, ckn, cd, ck,
            is_training, previous_model):

    # put N, K, alpha, beta in a K-length vector
    N_arr, K_arr, alpha_arr, beta_arr = _setup_arr(N, K, alpha, beta)

    all_lls = []
    thin = 0
    for samp in range(n_samples):
    
        s = samp+1        
        if s >= n_burn:
            print("Sample " + str(s) + " "),
        else:
            print("Burn-in " + str(s) + " "),
            
        for d in range(D):

            if d%10==0:                        
                sys.stdout.write('.')
                sys.stdout.flush()
            
            word_idx = document_indices[d]
            for pos, n in enumerate(word_idx):
                
                k = Z[(d, pos)]
                _nb_remove_from_model(k, cdk, cd, ck, ckn, d, pos, n)

                # compute log prior and log likelihood
                if is_training:
                    # log_likelihood = np.log(ckn[:, n] + beta) - np.log(ck + N*beta)
                    temp = ckn[:, n]
                    log_likelihood = _nb_compute_left(temp, ck, N_arr, beta_arr)
                else:
                    print("not done yet!")

                # log_prior = np.log(cdk[d, :] + alpha) - np.log(cd[d] + K*alpha)        
                temp = cdk[d, :]
                temp2 = np.empty(len(temp))
                temp2.fill(cd[d])                        
                temp2 = temp2.astype(np.int64)                                    
                log_prior = _nb_compute_right(temp, temp2, K_arr, alpha_arr)
                
                # log_post = log_likelihood + log_prior
                log_post = _nb_add(log_likelihood, log_prior)

                # post = np.exp(log_post - log_post.max())
                # post = post / post.sum()
                post = np.empty(K)
                _nb_normalise(log_post, post)
                                        
                # k = np.random.multinomial(1, post).argmax()
                cumsum = np.empty(K)
                _nb_cumsum(post, cumsum)
                random_number = np.random.rand()
                k = _nb_sample_index(cumsum, random_number)
         
                _nb_assign_to_model(cdk, cd, ck, ckn, d, k, pos, n)
                Z[(d, pos)] = k

        if s > n_burn:
            thin += 1
            if thin%n_thin==0:    
                ll = K * ( gammaln(N*beta) - (gammaln(beta)*N) )
                for k in range(K):
                    for n in range(N):
                        ll += gammaln(ckn[k, n]+beta)
                    ll -= gammaln(ck[k] + N*beta)
                ll += D * ( gammaln(K*alpha) - (gammaln(alpha)*K) )
                all_lls.append(ll)      
                print(" Log likelihood = %.3f " % ll)                        
            else:                
                print
        else:
            print
            
    # update phi
    phi = ckn + beta
    phi /= np.sum(phi, axis=1)[:, np.newaxis]

    # update theta
    theta = cdk + alpha 
    theta /= np.sum(theta, axis=1)[:, np.newaxis]

    all_lls = np.array(all_lls)            
    return phi, theta, all_lls

def _setup_arr(N, K, alpha, beta):
    
    N_arr = np.empty(K)
    K_arr = np.empty(K)            
    alpha_arr = np.empty(K)
    beta_arr = np.empty(K)            

    N_arr.fill(N)
    K_arr.fill(K)
    alpha_arr.fill(alpha)
    beta_arr.fill(beta)
    
    N_arr = N_arr.astype(np.int64)            
    K_arr = N_arr.astype(np.int64)            

    return N_arr, K_arr, alpha_arr, beta_arr

@jit(nopython=True)
def _nb_remove_from_model(k, cdk, cd, ck, ckn, d, pos, n):
    cdk[d, k] -= 1
    cd[d] -= 1                    
    ck[k] -= 1
    ckn[k, n] -= 1

@jit(nopython=True)
def _nb_assign_to_model(cdk, cd, ck, ckn, d, k, pos, n):
    cdk[d, k] += 1
    cd[d] += 1
    ck[k] += 1
    ckn[k, n] += 1

@vectorize([double(int64, int64, int64, double)], nopython=True)
def _nb_compute_left(ckn, ck, N, beta):
    log_likelihood = math.log(ckn + beta) - math.log(ck + N*beta)
    return log_likelihood                                

@vectorize([double(int64, int64, int64, double)], nopython=True)
def _nb_compute_right(cdk, cd, K, alpha):
    log_prior = math.log(cdk + alpha) - math.log(cd + K*alpha)
    return log_prior                                

@vectorize([double(double, double)], nopython=True)
def _nb_add(x, y):
    return x + y

@jit(nopython=True)
def _nb_max(arr):
    maxval = arr[0]
    for i in arr:
        if i > maxval:
            maxval = i
    return maxval

@jit(nopython=True)
def _nb_normalise(log_post, post):
    max_log_post = _nb_max(log_post)            
    sum_post = 0
    for i in range(len(log_post)):
        post[i] = math.exp(log_post[i] - max_log_post)
        sum_post += post[i]
    for i in range(len(post)):
        post[i] = post[i] / sum_post

@jit(nopython=True)
def _nb_cumsum(arr, cumsum):
    total = 0
    for i in range(len(arr)):
        val = arr[i]
        total += val
        cumsum[i] = total

@jit(nopython=True)
def _nb_sample_index(cumsum, random_number):
    k = 0
    for k in range(len(cumsum)):
        c = cumsum[k]
        if random_number <= c:
            break 
    return k