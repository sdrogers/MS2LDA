import math
import sys

from numba import jit
from numba.types import int64, float64

import numpy as np

def sample_numba(random_state, n_burn, n_samples, n_thin, 
            D, N, K, document_indices, vocab_type,
            alpha, beta, 
            Z, cdk, cd, previous_K, 
            bag_indices, bags):

    all_lls = []
    thin = 0
    
    # prepare some K-length vectors to hold the intermediate results during loop
    post = np.empty(K, dtype=np.float64)
    cumsum = np.empty(K, dtype=np.float64)

    # precompute repeated constants
    N_beta = N * beta
    K_alpha = K * alpha    

    # not sure if it's possible to pass a variable number of arguments to Numba's JIT'ed function??!!
    # so we create all these separate variables ...
    ckn0, ck0, previous_ckn0, previous_ck0 = _get_count_matrices(0, bags)
    ckn1, ck1, previous_ckn1, previous_ck1 = _get_count_matrices(1, bags)
    ckn2, ck2, previous_ckn2, previous_ck2 = _get_count_matrices(2, bags)

    # loop over samples
    for samp in range(n_samples):
    
        s = samp+1        
        if s >= n_burn:
            print("Sample " + str(s) + " "),
        else:
            print("Burn-in " + str(s) + " "),

        # loop over documents
        for d in range(D):

            if d%10==0:                        
                sys.stdout.write('.')
                sys.stdout.flush()

            # loop over words, not so easy to JIT due to rng and Z
            word_locs = document_indices[d]
            for pos, n in word_locs:
                random_number = random_state.rand()                
                k = Z[(d, pos)]         
                b = vocab_type[n]   
                k = _nb_get_new_index(d, n, k, cdk, cd, 
                                      N, K, previous_K, alpha, beta, 
                                      N_beta, K_alpha,
                                      post, cumsum, random_number, b,
                                      ckn0, ck0, previous_ckn0, previous_ck0,
                                      ckn1, ck1, previous_ckn1, previous_ck1,
                                      ckn2, ck2, previous_ckn2, previous_ck2)
                Z[(d, pos)] = k

        if s > n_burn:
        
            thin += 1
            if thin%n_thin==0:    

                ll = 0
                for bi in bag_indices:
                    ckn = bags[bi].ckn
                    ck = bags[bi].ck
                    beta_bi = beta[bi]
                    ll += _nb_p_w_z(N, K, beta_bi, ckn, ck)
                ll += _nb_p_z(D, K, alpha, cdk, cd)                  
                all_lls.append(ll)      
                print(" Log joint likelihood = %.3f " % ll)                        
            
            else:                
                print
        
        else:
            print
            
    # update phi
    phi = None
    for bi in bag_indices:            
        # update phi for each bag
        current_phi = bags[bi].ckn + beta[bi]
        current_phi /= np.sum(current_phi, axis=1)[:, np.newaxis]
        # accumulate the product
        if phi is None:
            phi = current_phi
        else:
            phi = np.multiply(phi, current_phi)

    # update theta
    theta = cdk + alpha 
    theta /= np.sum(theta, axis=1)[:, np.newaxis]

    all_lls = np.array(all_lls)            
    return phi, theta, all_lls

def _get_count_matrices(b, bags):
    ckn = bags[b].ckn
    ck = bags[b].ck
    previous_ckn = bags[b].previous_ckn
    previous_ck = bags[b].previous_ck
    return ckn, ck, previous_ckn, previous_ck

@jit(int64(
           int64, int64, int64, int64[:, :], int64[:], 
           int64, int64, int64, float64, float64[:],
           float64[:], float64,
           float64[:], float64[:], float64, int64,
           int64[:, :], int64[:], int64[:, :], int64[:],
           int64[:, :], int64[:], int64[:, :], int64[:],
           int64[:, :], int64[:], int64[:, :], int64[:]
), nopython=True)
def _nb_get_new_index(d, n, k, cdk, cd, 
                      N, K, previous_K, alpha, beta, 
                      N_beta, K_alpha,                      
                      post, cumsum, random_number, b,
                      ckn0, ck0, previous_ckn0, previous_ck0,
                      ckn1, ck1, previous_ckn1, previous_ck1,
                      ckn2, ck2, previous_ckn2, previous_ck2):

    temp_ckn0 = ckn0[:, n]
    temp_previous_ckn0 = previous_ckn0[:, n]
    temp_ckn1 = ckn1[:, n]
    temp_previous_ckn1 = previous_ckn1[:, n]
    temp_ckn2 = ckn2[:, n]
    temp_previous_ckn2 = previous_ckn2[:, n]
    
    temp_cdk = cdk[d, :]

    # remove from model
    cdk[d, k] -= 1
    cd[d] -= 1
    if b == 0: 
        ckn0[k, n] -= 1
        ck0[k] -= 1
    elif b == 1:
        ckn1[k, n] -= 1
        ck1[k] -= 1
    else:
        ckn2[k, n] -= 1
        ck2[k] -= 1        

    # compute likelihood, prior, posterior
    for i in range(len(post)):

        # we risk underflowing by not working in log space here
        if i < previous_K:
            temp0 = (temp_previous_ckn0[i] + beta[0]) / (previous_ck0[i] + N_beta[0])
            temp1 = (temp_previous_ckn1[i] + beta[1]) / (previous_ck1[i] + N_beta[1])
            temp2 = (temp_previous_ckn2[i] + beta[2]) / (previous_ck2[i] + N_beta[2])
            likelihood = temp0 * temp1 * temp2
        else:
            temp0 = (temp_ckn0[i] + beta[0]) / (ck0[i] + N_beta[0])
            temp1 = (temp_ckn1[i] + beta[1]) / (ck1[i] + N_beta[1])
            temp2 = (temp_ckn2[i] + beta[2]) / (ck2[i] + N_beta[2])
            likelihood = temp0 * temp1 * temp2
        prior = (temp_cdk[i] + alpha) / (cd[d] + K_alpha)
        post[i] = likelihood * prior

        # better but slower code
#         if i < previous_K:
#             temp0 = math.log(temp_previous_ckn0[i] + beta[0]) - math.log(previous_ck0[i] + N_beta[0])
#             temp1 = math.log(temp_previous_ckn1[i] + beta[1]) - math.log(previous_ck1[i] + N_beta[1])
#             temp2 = math.log(temp_previous_ckn2[i] + beta[2]) - math.log(previous_ck2[i] + N_beta[2])
#             likelihood = temp0 + temp1 + temp2
#         else:
#             temp0 = math.log(temp_ckn0[i] + beta[0]) - math.log(ck0[i] + N_beta[0])
#             temp1 = math.log(temp_ckn1[i] + beta[1]) - math.log(ck1[i] + N_beta[1])
#             temp2 = math.log(temp_ckn2[i] + beta[2]) - math.log(ck2[i] + N_beta[2])
#             likelihood = temp0 + temp1 + temp2
#         prior = math.log(temp_cdk[i] + alpha) - math.log(cd[d] + K_alpha)
#         post[i] = likelihood + prior

    # normalise posterior
    # we risk underflowing by not working in log space here    
    sum_post = 0
    for i in range(len(post)):
        sum_post += post[i]
    for i in range(len(post)):
        post[i] = post[i] / sum_post
    
    # better but slower code
#     max_log_post = post[0]
#     for i in range(len(post)):
#         val = post[i]
#         if val > max_log_post:
#             max_log_post = val
#     sum_post = 0
#     for i in range(len(post)):
#         post[i] = math.exp(post[i] - max_log_post)
#         sum_post += post[i]
#     for i in range(len(post)):
#         post[i] = post[i] / sum_post

    # sample k from multinomial                            
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
    if b == 0: 
        ckn0[k, n] += 1
        ck0[k] += 1
    elif b == 1:
        ckn1[k, n] += 1
        ck1[k] += 1
    else:
        ckn2[k, n] += 1
        ck2[k] += 1      
    
    return k

@jit(float64(int64, int64, float64, int64[:, :], int64[:]), nopython=True)
def _nb_p_w_z(N, K, beta, ckn, ck):
    ll = K * ( math.lgamma(N*beta) - (math.lgamma(beta)*N) )
    for k in range(K):
        for n in range(N):
            ll += math.lgamma(ckn[k, n]+beta)
        ll -= math.lgamma(ck[k] + N*beta)
    return ll

@jit(float64(int64, int64, float64, int64[:, :], int64[:]), nopython=True)
def _nb_p_z(D, K, alpha, cdk, cd):
    ll = D * ( math.lgamma(K*alpha) - (math.lgamma(alpha)*K) )
    for d in range(D):
        for k in range(K):
            ll += math.lgamma(cdk[d, k]+alpha)
        ll -= math.lgamma(cd[d] + K*alpha)
    return ll