import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
path = '/home/ubuntu/linux/Work/Deep_Learning/Amazon Access/Data/all/'
train = pd.read_csv(path+'train.csv')
test = pd.read_csv(path+'test.csv')

print 'Train and test are (%s %s)' %(train.shape[0],test.shape[0])

cols = train.columns
len(np.setdiff1d(test[cols[1]].unique(), train[cols[1]].unique()))
[len(np.setdiff1d(test[cols[elem]].unique(), train[cols[elem]].unique())) for elem in range(1,len(cols))]

train.groupby(cols[3])[cols[3]].count()

def counts_cats():
    final_counts = {}
    for elem in cols[2:]:
        final_counts[elem] = train.groupby(elem)[elem].count()

    return final_counts

final_counts = counts_cats()

temp = pd.DataFrame({cols[4]: final_counts[cols[4]]})
temp[cols[3]+'_tags'] = np.where(temp.ROLE_ROLLUP_1> 50, temp.index, 99999999)

plt.title(cols[5])
plt.hist(final_counts[cols[5]], bins = 100, range = (0,300))
plt.show()
