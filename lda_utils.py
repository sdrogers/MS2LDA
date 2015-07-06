import math

from scipy.special import psi, polygamma

import numpy as np


def word_indices(document):
    """
    Turns a document vector of word counts into a vector of the indices
     words that have non-zero counts, repeated for each count
    e.g. 
    >>> word_indices(np.array([3, 0, 1, 2, 0, 5]))
    [0, 0, 0, 2, 3, 3, 5, 5, 5, 5, 5]
    """
    results = []
    for nnz in document.values.nonzero()[1]:
        for n in range(int(document[nnz])):
            results.append(nnz)
    return results

def psi_inverse(initial_x, y, num_iter=5):
    """
    Computes the inverse digamma function using Newton's method
    See Appendix c of Minka, T. P. (2003). Estimating a Dirichlet distribution. 
    Annals of Physics, 2000(8), 1-13. http://doi.org/10.1007/s00256-007-0299-1 for details.
    """
    
    # initialisation
    if y >= -2.22:
        x_old = np.exp(y)+0.5
    else:
        gamma_val = -psi(1)
        x_old = -(1/(y+gamma_val))    

    # do Newton update here
    for i in range(num_iter):
        numerator = psi(x_old) - y
        denumerator = polygamma(1, x_old)
        x_new = x_old - (numerator/denumerator)
        x_old = x_new
    
    return x_new

def estimate_alpha_from_theta(D, K, initial_alpha, thetas, n_iter=100):
    """
    Estimate posterior alpha of a Dirichlet from multinomial vectors drawn from the Dirichlet
    see Huang, J. (2005). Maximum Likelihood Estimation of Dirichlet Distribution Parameters. Distribution, 40(2), 1-9. doi:10.1.1.93.2881
    """
     
    # initialise old and new alphas before iteration
    alpha_old = np.ones(K) * initial_alpha
    alpha_new = np.zeros_like(alpha_old)
    temp = np.log(thetas)
    log_pk = np.sum(temp, axis=0)
    log_pk /= D        
    
    for i in range(n_iter):
        
        # update concentration parameter of a dirichlet from samples of the multinomial vector theta
        psi_sum_alpha_old = psi(np.sum(alpha_old))
        for k in range(K):
            initial_x = alpha_old[k]
            y = psi_sum_alpha_old+log_pk[k]
            alpha_new[k] = psi_inverse(initial_x, y)
        
        # set alpha_new to alpha_old for the next iteration update
        alpha_old = alpha_new  

    return alpha_new

def estimate_alpha_from_counts(D, K, initial_alpha, counts, n_iter=1000000):

    counts = counts.astype(float)
    ni = np.sum(counts, axis=0)

    # initialise old and new alphas before iteration
    alpha_old = np.ones(K) * initial_alpha
    alpha_new = np.zeros_like(alpha_old)
    n_iter = 20
    for i in range(n_iter):

        for k in range(K):
             
            # compute numerator
            nik = counts[:, k]
            alpha_k = alpha_old[k]
            # numerator = np.sum(psi(nik+alpha_k)-psi(alpha_k))
            numerator = np.sum(nik/(nik-1+alpha_k)) # LOO approximation
             
            # compute denumerator
            sum_alpha = np.sum(alpha_old)
            # denumerator = np.sum(psi(ni+sum_alpha)-psi(sum_alpha))
            denumerator = np.sum(ni/(ni-1+sum_alpha)) # LOO approximation
             
            # compute new alpha[k]
            alpha_new[k] = alpha_old[k] * (numerator/denumerator)
        
        # set alpha_new to alpha_old for the next iteration update
        alpha_old = alpha_new    

    return alpha_new

def main():
    
    np.set_printoptions(suppress=True)
    
    print 'Test psi_inverse()'
    x = 0.84492
    y = psi(x)
    print 'x = ' + str(x)
    print 'y = ' + str(y)
    x_inverse = psi_inverse(0.2, y, num_iter=5)
    print 'x_inverse = ' + str(x_inverse)
    print
    
    print 'Test estimate_alpha()'
    D = 100
    K = 40
    alpha = 0.2
    alpha_vec = [alpha] * K
    print 'initial alpha = ' + str(alpha)

    thetas = np.empty((D, K))    
    counts = np.empty((D, K))
    for d in range(D):
        # sample topic proportions with uniform dirichlet parameter alpha of length n_topics
        theta = np.random.mtrand.dirichlet(alpha_vec)
        thetas[d, :] = theta
        # sample counts from theta
        counts[d, :] = np.random.multinomial(1000, theta, size=1)

    print
    print 'thetas'
    print thetas
    print

    print 'counts'
    print counts
    print

    alpha_hat = estimate_alpha_from_theta(D, K, alpha_vec, thetas)
    print 'estimated alpha from thetas = ' + str(alpha_hat)
    print

    alpha_hat = estimate_alpha_from_counts(D, K, alpha_vec, counts)
    print 'estimated alpha from counts = ' + str(alpha_hat)
    print

if __name__ == "__main__":
    main()