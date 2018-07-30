import pandas as pd
import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

with open ('/home/ubuntu/linux/Work/Deep_Learning/Amazon_Access/Subs/final_results.csv', 'wb') as f:
    with open('/home/ubuntu/pred_access_final.txt', 'rb') as file:
        for vals in file:
            vals = sigmoid(np.float(vals))
            f.write('%s\n'%(vals))

results = pd.read_csv('/home/ubuntu/linux/Work/Deep_Learning/Amazon_Access/Subs/final_results.csv',header= None)
final = pd.concat([test['id'],results], axis = 1)
final.rename(columns = {'id': 'Id', 0: 'Action'}, inplace = True)

final.to_csv('/home/ubuntu/linux/Work/Deep_Learning/Amazon_Access/Subs/subs.csv', index = False)