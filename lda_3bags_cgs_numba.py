import math
import sys

import numba as nb
from numba import jit
from numba.types import int64, float64

import numpy as np

bag_of_word_dtype = np.dtype([('bag1', np.int64),
                 ('bag2', np.int64),
                 ('bag3', np.int64)])
numba_bag_of_word_dtype = nb.from_dtype(bag_of_word_dtype)

def sample_numba(random_state, n_burn, n_samples, n_thin, 
            D, N, K, document_indices, vocab_type,
            alpha, beta, 
            Z, cdk, cd, previous_K, 
            bag_indices, bags):
    
    # prepare some K-length vectors to hold the intermediate results during loop
    post = np.empty(K, dtype=np.float64)
    cumsum = np.empty(K, dtype=np.float64)

    # precompute repeated constants
    N_beta = N * beta
    K_alpha = K * alpha    

    # each of this is a structured array of type bag_of_word
    ckn = np.zeros_like(bags[0].ckn, dtype=bag_of_word_dtype)
    ck = np.zeros_like(bags[0].ck, dtype=bag_of_word_dtype)
    previous_ckn = np.zeros_like(bags[0].previous_ckn, dtype=bag_of_word_dtype)
    previous_ck = np.zeros_like(bags[0].previous_ck, dtype=bag_of_word_dtype)

    # transfer the counts from the bags to the structured arrays
    _populate_count_matrices(bags, ckn, ck, previous_ckn, previous_ck)    
    
    # prepare the input matrices
    print "Preparing words"
    all_d = []
    all_pos = []
    all_n = []
    total_words = 0
    max_pos = 0
    for d in range(D):
        word_locs = document_indices[d]
        for pos, n in word_locs:
            total_words += 1
            all_d.append(d)
            all_pos.append(pos)
            all_n.append(n)
            if pos > max_pos:
                max_pos = pos
    all_d = np.array(all_d, dtype=np.int64)
    all_pos = np.array(all_pos, dtype=np.int64)
    all_n = np.array(all_n, dtype=np.int64)
    vocab_type = np.array(vocab_type, dtype=np.int)
    
    print "Preparing Z matrix"    
    Z_mat = np.empty((D, max_pos+1), dtype=np.int64)
    for d in range(D):
        word_locs = document_indices[d]
        for pos, n in word_locs:
            k = Z[(d, pos)]
            Z_mat[d, pos] = k
    
    print "DONE"

    # loop over samples
    all_lls = []
    thin = 0
    for samp in range(n_samples):

        s = samp+1        
        if s >= n_burn:
            print("Sample " + str(s) + " "),
        else:
            print("Burn-in " + str(s) + " "),    

        all_random = random_state.rand(total_words)
        ll = _nb_do_sampling(s, n_burn, total_words, all_d, all_pos, all_n, all_random, Z_mat, vocab_type,
                          cdk, cd, 
                          D, N, K, previous_K, alpha, beta, 
                          N_beta, K_alpha,                      
                          post, cumsum,
                          ckn, ck, previous_ckn, previous_ck)
        if s > n_burn:        
            thin += 1
            if thin%n_thin==0:    
                all_lls.append(ll)      
                print(" Log joint likelihood = %.3f " % ll)
            else:                
                print
        else:
            print
                
    # update phi
    phi1 = ckn['bag1'] + beta[0]
    phi1 /= np.sum(phi1, axis=1)[:, np.newaxis]
    phi2 = ckn['bag2'] + beta[1]
    phi2 /= np.sum(phi2, axis=1)[:, np.newaxis]
    phi3 = ckn['bag3'] + beta[2]
    phi3 /= np.sum(phi3, axis=1)[:, np.newaxis]
    phis = [phi1, phi2, phi3]

    # update theta
    theta = cdk + alpha 
    theta /= np.sum(theta, axis=1)[:, np.newaxis]

    all_lls = np.array(all_lls)            
    return phis, theta, all_lls

def _populate_count_matrices(bags, ckn, ck, previous_ckn, previous_ck):
    ckn['bag1'] = bags[0].ckn
    ckn['bag2'] = bags[1].ckn
    ckn['bag3'] = bags[2].ckn
    ck['bag1'] = bags[0].ck
    ck['bag2'] = bags[1].ck
    ck['bag3'] = bags[2].ck
    previous_ckn['bag1'] = bags[0].previous_ckn
    previous_ckn['bag2'] = bags[1].previous_ckn
    previous_ckn['bag3'] = bags[2].previous_ckn
    previous_ck['bag1'] = bags[0].previous_ck
    previous_ck['bag2'] = bags[1].previous_ck
    previous_ck['bag3'] = bags[2].previous_ck

@jit(int64(
           int64, int64, int64, int64[:, :], int64[:], 
           int64, int64, int64, float64, float64[:],
           float64[:], float64,
           float64[:], float64[:], float64, int64,
           numba_bag_of_word_dtype[:, :], numba_bag_of_word_dtype[:], numba_bag_of_word_dtype[:, :], numba_bag_of_word_dtype[:]
), nopython=True)
def _nb_get_new_index(d, n, k, cdk, cd, 
                      N, K, previous_K, alpha, beta, 
                      N_beta, K_alpha,                      
                      post, cumsum, random_number, b,
                      ckn, ck, previous_ckn, previous_ck):

    temp_ckn = ckn[:, n]
    temp_previous_ckn = previous_ckn[:, n]    
    temp_cdk = cdk[d, :]

    # remove from model
    cdk[d, k] -= 1
    cd[d] -= 1
    if b == 0: 
        ckn[k, n].bag1 -= 1
        ck[k].bag1 -= 1
    elif b == 1:
        ckn[k, n].bag2 -= 1
        ck[k].bag2 -= 1
    else:
        ckn[k, n].bag3 -= 1
        ck[k].bag3 -= 1        

    # compute likelihood, prior, posterior
    for i in range(len(post)):

        # we risk underflowing by not working in log space here
        if i < previous_K:
            temp0 = (temp_previous_ckn[i].bag1 + beta[0]) / (previous_ck[i].bag1 + N_beta[0])
            temp1 = (temp_previous_ckn[i].bag2 + beta[1]) / (previous_ck[i].bag2 + N_beta[1])
            temp2 = (temp_previous_ckn[i].bag3 + beta[2]) / (previous_ck[i].bag3 + N_beta[2])
            likelihood = temp0 * temp1 * temp2
        else:
            temp0 = (temp_ckn[i].bag1 + beta[0]) / (ck[i].bag1 + N_beta[0])
            temp1 = (temp_ckn[i].bag2 + beta[1]) / (ck[i].bag2 + N_beta[1])
            temp2 = (temp_ckn[i].bag3 + beta[2]) / (ck[i].bag3 + N_beta[2])
            likelihood = temp0 * temp1 * temp2
        prior = (temp_cdk[i] + alpha) / (cd[d] + K_alpha)
        post[i] = likelihood * prior

        # better but slower code
#         if i < previous_K:
#             temp0 = math.log(temp_previous_ckn[i].bag1 + beta[0]) - math.log(previous_ck[i].bag1 + N_beta[0])
#             temp1 = math.log(temp_previous_ckn[i].bag2 + beta[1]) - math.log(previous_ck[i].bag2 + N_beta[1])
#             temp2 = math.log(temp_previous_ckn[i].bag3 + beta[2]) - math.log(previous_ck[i].bag3 + N_beta[2])
#             likelihood = temp0 + temp1 + temp2
#         else:
#             temp0 = math.log(temp_ckn[i].bag1 + beta[0]) - math.log(ck[i].bag1 + N_beta[0])
#             temp1 = math.log(temp_ckn[i].bag2 + beta[1]) - math.log(ck[i].bag2 + N_beta[1])
#             temp2 = math.log(temp_ckn[i].bag3 + beta[2]) - math.log(ck[i].bag3 + N_beta[2])
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
        ckn[k, n].bag1 += 1
        ck[k].bag1 += 1
    elif b == 1:
        ckn[k, n].bag2 += 1
        ck[k].bag2 += 1
    else:
        ckn[k, n].bag3 += 1
        ck[k].bag3 += 1      
    
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
  
# @jit(int64(int64, int64, int64, int64, int64[:], int64[:], int64[:], float64[:, :], int64[:, :], int64[:], int64[:],
#            int64[:, :], int64[:], 
#            int64, int64, int64, int64, float64, float64[:],
#            float64[:], float64,
#            float64[:], float64[:],
#            numba_bag_of_word_dtype[:, :], numba_bag_of_word_dtype[:], numba_bag_of_word_dtype[:, :], numba_bag_of_word_dtype[:]
# ), nopython=True)   
@jit(nopython=True)   
def _nb_do_sampling(s, n_burn, total_words, all_d, all_pos, all_n, all_random, Z_mat, vocab_type,
                      cdk, cd, 
                      D, N, K, previous_K, alpha, beta, 
                      N_beta, K_alpha,                      
                      post, cumsum,
                      ckn, ck, previous_ckn, previous_ck):
    
    # loop over documents and all words in the document
    for w in range(total_words):
        
        d = all_d[w]
        pos = all_pos[w]
        n = all_n[w]
        random_number = all_random[w]
    
        # assign new k
        k = Z_mat[d, pos]         
        b = vocab_type[n]   
        k = _nb_get_new_index(d, n, k, cdk, cd, 
                              N, K, previous_K, alpha, beta, 
                              N_beta, K_alpha,
                              post, cumsum, random_number, b,
                              ckn, ck, previous_ckn, previous_ck)
        Z_mat[d, pos] = k

    ll = 0
    if s > n_burn:    
        ll += _nb_p_w_z(N, K, beta[0], ckn.bag1, ck.bag1)
        ll += _nb_p_w_z(N, K, beta[1], ckn.bag2, ck.bag2)
        ll += _nb_p_w_z(N, K, beta[2], ckn.bag3, ck.bag3)                
        ll += _nb_p_z(D, K, alpha, cdk, cd)                          

    return ll