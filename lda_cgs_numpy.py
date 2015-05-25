import sys

from scipy.special import gammaln

import numpy as np


def sample_numpy(random_state, n_burn, n_samples, n_thin, 
            D, N, K, document_indices, 
            alpha, beta, 
            Z, cdk, ckn, cd, ck,
            previous_ckn, previous_ck, silent, cv):

    all_lls = []
    thin = 0
    N_beta = N * beta
    K_alpha = K * alpha    
    for samp in range(n_samples):
    
        s = samp+1        
        if not silent:
            if s >= n_burn:
                print("Sample " + str(s) + " "),
            else:
                print("Burn-in " + str(s) + " "),
            
        for d in range(D):

            if not silent and d%10==0:                        
                sys.stdout.write('.')
                sys.stdout.flush()
            
            word_locs = document_indices[d]
            for pos, n in word_locs:
                
                # remove word from model
                k = Z[(d, pos)]
                cdk[d, k] -= 1
                cd[d] -= 1                    
                ck[k] -= 1
                ckn[k, n] -= 1

                # compute log prior and log likelihood
                log_likelihood = np.log(ckn[:, n] + previous_ckn[:, n] + beta) - \
                    np.log(ck + previous_ck + N_beta)            
                log_prior = np.log(cdk[d, :] + alpha) - np.log(cd[d] + K_alpha)        
                
                # sample new k from the posterior distribution log_post
                log_post = log_likelihood + log_prior
                post = np.exp(log_post - log_post.max())
                post = post / post.sum()
                k = random_state.multinomial(1, post).argmax()
         
                # reassign word back into model
                cdk[d, k] += 1
                cd[d] += 1
                ck[k] += 1
                ckn[k, n] += 1
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
