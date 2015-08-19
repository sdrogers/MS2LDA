import sys

from scipy.special import gammaln

import numpy as np
from mixture_cgs import Sample

def sample_numpy(random_state, n_burn, n_samples, n_thin, 
            D, N, K, document_indices, 
            alpha, beta, 
            Z, cdk, ckn, ck):

    samples = []
    all_lls = []
    thin = 0
    N_beta = np.sum(beta)
    K_alpha = np.sum(alpha)        
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
            
            # remove document from cluster k
            k = Z[d]
            cdk[k] -= 1                
            
            # remove counts for all the words in the document from cluster k
            word_locs = document_indices[d]
            for pos, n in word_locs:
                ckn[k, n] -= 1
                ck[k] -= 1

            # compute likelihood of document in new cluster    
            log_prior = np.log(cdk + alpha)                
            log_likelihood = np.zeros_like(log_prior)
            for k in range(K):
                for pos, n in word_locs:
                    log_likelihood[k] += np.log(ckn[k, n] + beta[n]) - np.log(ck[k] + N_beta)

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
         
            # reassign document back into model
            for pos, n in word_locs:
                ckn[k, n] += 1
                ck[k] += 1
            cdk[k] += 1
            Z[d] = k

        ll = K * ( gammaln(N_beta) - np.sum(gammaln(beta)) )
        for k in range(K):
            for n in range(N):
                ll += gammaln(ckn[k, n]+beta[n])
            ll -= gammaln(ck[k] + N_beta)                        

        ll += gammaln(K_alpha) - np.sum(gammaln(alpha))
        for k in range(K):
            ll += gammaln(cdk[k]+alpha[k])
        ll -= gammaln(np.sum(cdk) + K_alpha)                
            
        all_lls.append(ll)      
        print(" Log likelihood = %.3f " % ll)     

        if s > n_burn:
            thin += 1
            if thin%n_thin==0:                    
                cdk_copy = np.copy(cdk)
                ckn_copy = np.copy(ckn)
                to_store = Sample(cdk_copy, ckn_copy)
                samples.append(to_store)
                                   
            else:                
                print
        else:
            print
            
    all_lls = np.array(all_lls)
    return all_lls, samples    