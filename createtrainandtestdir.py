import shutil
from random import shuffle
from collections import Counter
from pprint import pprint
import pandas as pd
import os

r = list(range(2137))
shuffle(r)

filedata = pd.read_csv('namevsspeices.csv')
train = filedata.loc[r[:1500]]
test = filedata.loc[r[1501:]]
print(train.groupby('speices').count())
print(test.groupby('speices').count())



try:
    shutil.rmtree('train/')
except shutil.Error as err:
    print(err)
    
os.mkdir('train')    
    
for i in train['name']:
    print(str(i))
    shutil.copy('signal/'+i,'train')
train.to_csv('train/train_signal.csv')


try:
    shutil.rmtree('test/')
except shutil.Error as err:
    print(err)    
os.mkdir('test')

for i in test['name']:
    shutil.copy('signal/' + i,'test')    
test.to_csv('test/test_signal.csv')
