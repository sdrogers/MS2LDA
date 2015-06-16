import sys

from scipy.special import gammaln

import numpy as np


def sample_numpy(random_state, n_burn, n_samples, n_thin, 
            D, N, K, document_indices, vocab_type,
            alpha, beta, 
            Z, cdk, cd, previous_K, 
            bag_indices, bags):

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
                
                # remove word from model
                k = Z[(d, pos)]
                cdk[d, k] -= 1
                cd[d] -= 1    
                bags[b].ckn[k, n] -= 1
                bags[b].ck[k] -= 1

                log_prior = np.log(cdk[d, :] + alpha) - np.log(cd[d] + K_alpha)
                log_likelihood = np.zeros_like(log_prior)

                if previous_K == 0:

                    # for training
                    for bi in bag_indices:
                        log_likelihood += np.log(bags[bi].ckn[:, n] + beta[bi]) - np.log(bags[bi].ck + N_beta[bi])
                                
                else:
                    
                    log_likelihood = 0
                    for bi in bag_indices:
                    
                        # for testing on unseen data
                        log_likelihood_previous = np.log(bags[bi].previous_ckn[:, n] + beta[bi]) - np.log(bags[bi].previous_ck + N_beta[bi])
                        log_likelihood_current = np.log(bags[bi].ckn[:, n] + beta[bi]) - np.log(bags[bi].ck + N_beta[bi])    
    
                        # The combined likelihood: 
                        # front is from previous topic-word distribution
                        # back is from current topic-word distribution
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
                bags[b].ckn[k, n] += 1
                bags[b].ck[k] += 1
                Z[(d, pos)] = k

        if s > n_burn:
        
            thin += 1
            if thin%n_thin==0:    

                for bi in bag_indices:
                    ll = K * ( gammaln(N*beta[bi]) - (gammaln(beta[bi])*N) )
                    for k in range(K):
                        for n in range(N):
                            ll += gammaln(bags[bi].ckn[k, n]+beta[bi])
                        ll -= gammaln(bags[bi].ck[k] + N*beta[bi])    
                        
                ll += D * ( gammaln(K*alpha) - (gammaln(alpha)*K) )
                for d in range(D):
                    for k in range(K):
                        ll += gammaln(cdk[d, k]+alpha)
                    ll -= gammaln(cd[d] + K*alpha)                                            

                all_lls.append(ll)      
                print(" Log joint likelihood = %.3f " % ll)                        
            
            else:                
                print
        
        else:
            print

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