import pandas as pd
import numpy as np
import pylab as plt

df = pd.read_csv('results.csv', sep=',|=', header=None)
df = df.sort([1])
print df

ks = df.ix[:, 1]
mean_margs = df.ix[:, 3]
mean_perplexities = df.ix[:, 5]

plt.figure()

plt.subplot(1, 2, 1)
plt.plot(np.array(ks), np.array(mean_margs))
plt.grid()
plt.xlabel('K')
plt.ylabel('Log evidence')
plt.title('Log evidence')

plt.subplot(1, 2, 2)
plt.plot(np.array(ks), np.array(mean_perplexities))
plt.grid()
plt.xlabel('K')
plt.ylabel('Perplexity')
plt.title('Perplexity')

plt.suptitle('CV Results')
plt.savefig('results.png')
plt.show()
