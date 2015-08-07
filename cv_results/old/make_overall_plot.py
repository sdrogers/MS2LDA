import pandas as pd
import numpy as np
import pylab as plt

def load_results(filename):
    df = pd.read_csv(filename, sep=',|=', header=None)
    df = df.sort([1])
    print "Loading " + filename
    print df
    ks = df.ix[:, 1]
    mean_margs = df.ix[:, 3]
    mean_perplexities = df.ix[:, 5]
    return ks, mean_margs, mean_perplexities

ks1, mean_margs1, mean_perplexities1 = load_results('beer3pos.is/results.csv')
ks2, mean_margs2, mean_perplexities2 = load_results('beer3pos.is.3bags/results.csv')
ks3, mean_margs3, mean_perplexities3 = load_results('urine37pos.is/results.csv')
ks4, mean_margs4, mean_perplexities4 = load_results('urine37pos.is.3bags/results.csv')

plt.figure()

plt.subplot(2, 2, 1)
plt.plot(np.array(ks1), np.array(mean_margs1), 'r', label='LDA')
plt.plot(np.array(ks2), np.array(mean_margs2), 'b', label='3bags-LDA')
plt.grid()
plt.xlabel('K')
plt.ylabel('Log evidence')
plt.legend(loc='lower right', shadow=True)
plt.title('Beer3Pos Log evidence')

plt.subplot(2, 2, 2)
plt.plot(np.array(ks1), np.array(mean_perplexities1), 'r', label='LDA')
plt.plot(np.array(ks1), np.array(mean_perplexities2), 'b', label='3bags-LDA')
plt.grid()
plt.xlabel('K')
plt.ylabel('Perplexity')
plt.legend(loc='upper right', shadow=True)
plt.title('Beer3Pos Perplexity')

plt.subplot(2, 2, 3)
plt.plot(np.array(ks3), np.array(mean_margs3), 'r', label='LDA')
plt.plot(np.array(ks4), np.array(mean_margs4), 'b', label='MS2-LDA')
plt.grid()
plt.xlabel('K')
plt.ylabel('Log evidence')
plt.legend(loc='lower right', shadow=True)
plt.title('Urine37Pos Log evidence')

plt.subplot(2, 2, 4)
plt.plot(np.array(ks3), np.array(mean_perplexities3), 'r', label='LDA')
plt.plot(np.array(ks4), np.array(mean_perplexities4), 'b', label='MS2-LDA')
plt.grid()
plt.xlabel('K')
plt.ylabel('Perplexity')
plt.legend(loc='upper right', shadow=True)
plt.title('Urine37Pos Perplexity')

plt.tight_layout()
plt.suptitle('CV Results')
plt.savefig('results.png')
plt.show()
