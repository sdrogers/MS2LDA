from datetime import datetime
import math

from numba import jit
from numba.decorators import autojit
from numba.npyufunc.decorators import vectorize
from numba.types import int32, double, float_, int64

import numpy as np

@vectorize([double(int64, int64, int64, double)], nopython=True)
def ll(ckn, ck, N, beta):
    log_likelihood = math.log(ckn + beta) - math.log(ck + N*beta)
    return log_likelihood                                

def log_likelihood_python(S, ckn, ck, N, beta):

    N_arr = np.empty(len(ck))
    N_arr.fill(N)
    N_arr = N_arr.astype(np.int64)
    
    beta_arr = np.empty(len(ck))
    beta_arr.fill(beta)
    
    for s in range(S):
        print s
        for n in range(N):
            temp = ckn[:, n]
            log_likelihood = ll(temp, ck, N_arr, beta_arr)

def log_likelihood_numpy(S, ckn, ck, N, beta):
    for s in range(S):
        print s
        for n in range(N):
            log_likelihood = np.log(ckn[:, n] + beta) - np.log(ck + N*beta)

S = 10
K = 250
N = 20000
ckn = np.random.random_integers(0, high=500, size=(K, N))
ck = np.sum(ckn, axis=1)
beta = 0.01

startTime = datetime.now()
print "Running JIT log_likelihood_python()"
log_likelihood_python(S, ckn, ck, N, beta)
print "Total time = " + str(datetime.now()-startTime)
print

startTime = datetime.now()
print "Running log_likelihood_numpy()"
log_likelihood_numpy(S, ckn, ck, N, beta)
print "Total time = " + str(datetime.now()-startTime)
print