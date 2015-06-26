import sys

from scipy.special import gammaln

import numpy as np


def sample_numpy(random_state, n_burn, n_samples, n_thin, 
            D, N, K, document_indices, 
            alpha, beta, 
            Z, cdk, cd, previous_K,
            ckn, ck, previous_ckn, previous_ck, 
            vocab_type, bag_labels):    

    all_lls = []
    thin = 0
    N_beta = N * beta
    K_alpha = K * alpha    
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
            
            word_locs = document_indices[d]
            for pos, n in word_locs:
                
                b = vocab_type[n]
                initial_bag_label = bag_labels[b]
                
                # remove word from model
                k = Z[(d, pos)]
                cdk[d, k] -= 1
                cd[d] -= 1    
                bag_ckn = ckn[initial_bag_label]
                bag_ck = ck[initial_bag_label]
                bag_ckn[k, n] -= 1
                bag_ck[k] -= 1

                log_prior = np.log(cdk[d, :] + alpha) - np.log(cd[d] + K_alpha)
                log_likelihood = np.zeros_like(log_prior)

                if previous_K == 0:

                    # for training
                    for bi in range(len(bag_labels)):
                        bag_label = bag_labels[bi]
                        bag_ckn = ckn[bag_label]
                        bag_ck = ck[bag_label]
                        log_likelihood += np.log(bag_ckn[:, n] + beta[bi]) - np.log(bag_ck + N_beta[bi])
                                
                else:
                    
                    # for testing on unseen data
                    log_likelihood = 0
                    for bi in range(len(bag_labels)):

                        bag_label = bag_labels[bi]
                        bag_previous_ckn = previous_ckn[bag_label]
                        bag_previous_ck = previous_ck[bag_label]
                        bag_ckn = ckn[bag_label]
                        bag_ck = ck[bag_label]
                    
                        log_likelihood_previous = np.log(bag_previous_ckn[:, n] + beta[bi]) - np.log(bag_previous_ck + N_beta[bi])
                        log_likelihood_current = np.log(bag_ckn[:, n] + beta[bi]) - np.log(bag_ck + N_beta[bi])    

                        # The combined likelihood: 
                        # front is from previous topic-word distribution
                        # back is from current topic-word distribution
                        # Because of the values from the hyperparameter, we cannot do
                        # log_likelihood += log_likelihood_previous + log_likelihood_current  
                        front = log_likelihood_previous[0:previous_K]
                        back = log_likelihood_current[previous_K:]
                        log_likelihood += np.hstack((front, back))
                            
                # sample new k from the posterior distribution log_post
                log_post = log_likelihood + log_prior
                post = np.exp(log_post - log_post.max())
                post = post / post.sum()
                
                # k = random_state.multinomial(1, post).argmax()
                cumsum = np.empty(K, dtype=np.float64)
                random_number = random_state.rand()                                
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
         
                # reassign word back into model
                cdk[d, k] += 1
                cd[d] += 1
                bag_ckn = ckn[initial_bag_label]
                bag_ck = ck[initial_bag_label]
                bag_ckn[k, n] += 1
                bag_ck[k] += 1
                Z[(d, pos)] = k

        if s > n_burn:
        
            thin += 1
            if thin%n_thin==0:    
                
                ll = 0
                for bi in range(len(bag_labels)):
                    bag_label = bag_labels[bi]
                    bag_ckn = ckn[bag_label]
                    bag_ck = ck[bag_label]
                    beta_bi = beta[bi]
                    ll += p_w_z(N, K, beta_bi, bag_ckn, bag_ck)
                ll += p_z(D, K, alpha, cdk, cd)                  
                all_lls.append(ll)      
                print(" Log joint likelihood = %.3f " % ll)                                          
            
            else:                
                print
        
        else:
            print

    # update phi
    phis = []
    for bi in range(len(bag_labels)):  
        bag_label = bag_labels[bi]         
        bag_ckn = ckn[bag_label]
        phi = bag_ckn + beta[bi]
        phi /= np.sum(phi, axis=1)[:, np.newaxis]
        phis.append(phi)
        
    # update theta
    theta = cdk + alpha 
    theta /= np.sum(theta, axis=1)[:, np.newaxis]

    all_lls = np.array(all_lls)
    return phis, theta, all_lls

def p_w_z(N, K, beta, ckn, ck):
    val = K * ( gammaln(N*beta) - (gammaln(beta)*N) )
    for k in range(K):
        for n in range(N):
            val += gammaln(ckn[k, n]+beta)
        val -= gammaln(ck[k] + N*beta)      
    return val

def p_z(D, K, alpha, cdk, cd):
    val = D * ( gammaln(K*alpha) - (gammaln(alpha)*K) )
    for d in range(D):
        for k in range(K):
            val += gammaln(cdk[d, k]+alpha)
        val -= gammaln(cd[d] + K*alpha)      
    return val