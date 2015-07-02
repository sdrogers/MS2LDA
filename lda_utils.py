import numpy as np
from scipy.special import psi, polygamma

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

def main():
    
    x = 0.84492
    y = psi(x)
    print 'x = ' + str(x)
    print 'y = ' + str(y)
    x_inverse = psi_inverse(0.2, y, num_iter=5)
    print 'x_inverse = ' + str(x_inverse)

if __name__ == "__main__":
    main()