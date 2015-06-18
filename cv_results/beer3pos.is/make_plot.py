import pandas as pd
import numpy as np
import pylab as plt

df = pd.read_csv('results.csv', sep=',|=', header=None)
df = df.sort([1])
print df

X = df.ix[:, 1]
Y = df.ix[:, 3]
plt.plot(X, Y, linewidth=2, label='log marginal likelihood')
plt.legend(loc='lower right')
plt.grid()
plt.xlabel('K')
plt.ylabel('marg')
plt.title('Cross-validation on No. of topics -- beer3 pos')
plt.tight_layout()
plt.savefig('results.png')
plt.show()
