import pandas as pd
import numpy as np
import pylab as plt

def load_results(filename):
    df = pd.read_csv(filename, sep=',|=', header=None)
    df = df.sort([1])
    print "Loading " + filename
    print df
    ks = df.ix[:, 1]
    lda_margs = df.ix[:, 15]
    lda_perplexities = df.ix[:, 17]
    mixture_margs = df.ix[:, 19]
    mixture_perplexities = df.ix[:, 21]
    return ks, lda_margs, lda_perplexities, mixture_margs, mixture_perplexities

ks1, lda_margs1, lda_perplexities1, mixture_margs1, mixture_perplexities1 = load_results('beer3pos/results.csv')
ks2, lda_margs2, lda_perplexities2, mixture_margs2, mixture_perplexities2 = load_results('urine37pos/results.csv')

plt.figure()

plt.subplot(2, 2, 1)
plt.plot(np.array(ks1), np.array(lda_margs1), 'r', label='LDA')
plt.plot(np.array(ks1), np.array(mixture_margs1), 'b', label='Mixture')
plt.grid()
plt.xlabel('K')
plt.ylabel('Log evidence')
plt.legend(loc='lower right', shadow=True)
plt.title('Beer3Pos Log evidence')

plt.subplot(2, 2, 2)
plt.plot(np.array(ks1), np.array(lda_perplexities1), 'r', label='LDA')
plt.plot(np.array(ks1), np.array(mixture_perplexities1), 'b', label='Mixture')
plt.grid()
plt.xlabel('K')
plt.ylabel('Perplexity')
plt.legend(loc='upper right', shadow=True)
plt.title('Beer3Pos Perplexity')

plt.subplot(2, 2, 3)
plt.plot(np.array(ks2), np.array(lda_margs2), 'r', label='LDA')
plt.plot(np.array(ks2), np.array(mixture_margs2), 'b', label='Mixture')
plt.grid()
plt.xlabel('K')
plt.ylabel('Log evidence')
plt.legend(loc='lower right', shadow=True)
plt.title('Urine37Pos Log evidence')

plt.subplot(2, 2, 4)
plt.plot(np.array(ks2), np.array(lda_perplexities2), 'r', label='LDA')
plt.plot(np.array(ks2), np.array(mixture_perplexities2), 'b', label='Mixture')
plt.grid()
plt.xlabel('K')
plt.ylabel('Perplexity')
plt.legend(loc='upper right', shadow=True)
plt.title('Urine37Pos Perplexity')

plt.tight_layout()
plt.suptitle('CV Results')
plt.savefig('results.png')
plt.show()