import pandas as pd
import numpy as np
from csv import DictReader
from matplotlib import pyplot as plt
path = '/home/ubuntu/linux/Work/Deep_Learning/Amazon_Access/Data/all/'
#train = pd.read_csv(path+'train.csv')
test = pd.read_csv(path+'test.csv')

print 'Train and test are (%s %s)' %(train.shape[0],test.shape[0])

cols_namespace = [ elem for elem in train.columns if elem not in ['ACTION', 'ROLE_CODE']]
namespace = {value: 'e%s' %(index) for index, value  in enumerate(cols_namespace)}


def python_to_vowpal(path, outfile_name,path1,infile_name,train = True):
    with open(path1+outfile_name, 'wb') as outfile:
        for index, row in enumerate(DictReader(open(path+ infile_name +'.csv', 'rb'))):
            features = ""
            for k,v in row.items():
                if (k not in ["ACTION", "ROLE_CODE", "id"] and len(str(v)))> 0:
                    #features += " %s: _%s" %(k,v)
                    features += "|%s _%s " %(namespace[k],v)
            if train:
                if row['ACTION'] == '1':
                    label = 1
                    #importance = 100
                else:
                    label = -1
                    #importance = 1
                outfile.write("%s %s\n" %(label,features))
            else:
                outfile.write("1 %s\n" % (features))

################Function call##################

python_to_vowpal(path ='/home/ubuntu/linux/Work/Deep_Learning/Amazon Access/Data/all/', outfile_name = 'python_to_vw_train.vw',\
                 path1 = '/home/ubuntu/linux/Work/Deep_Learning/Amazon Access/VW_Sets/', \
                 infile_name = 'train', train=True)


python_to_vowpal(path ='/home/ubuntu/linux/Work/Deep_Learning/Amazon_Access/Data/all/', outfile_name = 'python_to_vw_test.vw',\
                 path1 = '/home/ubuntu/linux/Work/Deep_Learning/Amazon_Access/VW_Sets/', \
                 infile_name = 'test', train=False)

