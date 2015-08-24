import math

from scipy.special import psi, polygamma

import numpy as np
import scipy.io as sio

# http://stackoverflow.com/questions/2272149/round-to-5-or-other-number-in-python        
def round_nicely(x, base=5):
    return int(base * round(float(x)/base))
    
def threshold_matrix(matrix, epsilon=0.0):
    thresholded = matrix.copy()
    n_row, n_col = thresholded.shape
    for i in range(n_row):
        row = thresholded[i, :]
        if epsilon > 0:
            small = row < epsilon
            row[small] = 0
        else:
            smallest_val = np.min(row)
            smallest_arr = np.ones_like(row) * smallest_val
            close = np.isclose(row, smallest_arr)
            row[close] = 0        
    return thresholded

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
        count = document.values[0].flatten()
        for n in range(int(count[nnz])):
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
    Estimate posterior alpha of a Dirichlet multinomial from samples of the multinomial vectors.
    This implements the Newton's method as described in Huang, J. (2005). Maximum Likelihood Estimation of 
    Dirichlet Distribution Parameters. Distribution, 40(2), 1-9. doi:10.1.1.93.2881
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

def generate_counts_from_matlab(matfile):    
    mat_contents = sio.loadmat(matfile)
    a = mat_contents['a'][0]
    data = mat_contents['data']
    return a, data

def estimate_alpha_from_counts(D, K, initial_alpha, counts, n_iter=1000):
    """
    Estimate posterior alpha of a Dirichlet multinomial from samples of the multinomial counts.
    This implements the fixed point update as described in Minka, T. P. (2003). Estimating a Dirichlet distribution. 
    Annals of Physics, 2000(8), 1-13. http://doi.org/10.1007/s00256-007-0299-1
    """

    counts = counts.astype(float)
    sdata = np.sum(counts, axis=1)

    # initialise old and new alphas before iteration
    alpha_old = np.ones(K) * initial_alpha
    for i in range(n_iter):

        sa = np.sum(alpha_old)
        temp = np.tile(alpha_old, (D, 1))
        g = np.sum(psi(counts + temp), axis=0) - D*psi(alpha_old)
        h = np.sum(psi(sdata + sa)) - D*psi(sa)
        alpha_new = alpha_old * (g/h)
        if np.max(np.abs(alpha_new-alpha_old)) < 1e-6:
            break
        
        # set alpha_new to alpha_old for the next iteration update
        alpha_old = alpha_new    

    return alpha_new

def main():
    
    np.set_printoptions(suppress=True)

    ### test psi inverse function ###
    
    print 'Test psi_inverse()'
    x = 0.84492
    y = psi(x)
    print 'x = ' + str(x)
    print 'y = ' + str(y)
    x_inverse = psi_inverse(0.2, y, num_iter=5)
    print 'x_inverse = ' + str(x_inverse)
    print

    ### generate some synthetic data ###
    
    print 'Test estimate_alpha()'
    D = 100
    K = 40
    alpha = 0.2
    alpha_vec = np.array([alpha] * K)
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
    
    #### load from matlab to compare against polya_fit_simple.m in fastfit ####

    a, data = generate_counts_from_matlab('/home/joewandy/fastfit/fastfit/test.mat')
    D, K = data.shape
    alpha_hat = estimate_alpha_from_counts(D, K, a, data)
    print 'estimated alpha from counts from matlab = ' + str(alpha_hat)
    print

if __name__ == "__main__":
    main()