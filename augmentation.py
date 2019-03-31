import numpy as np
import pandas as pd
import os
from skimage.io import imread,imsave
import random
meta_train = pd.read_csv('train/train_signal.csv').sort_values(by = 'speices')
speices = pd.unique(meta_train['speices'])
nl = os.listdir('noise')
spl = []
nml = []
z=0
for spc in speices:
	z += 1
	
	x = meta_train[meta_train['speices'] == spc]
	threshold = 25# As you wish
	p = 0
	if len(x)>threshold:		
		print(z)
		y = x.sample(threshold)
		for i in y.name:
			for j in y.name:
				a = random.random()
				b = .4			
				p += 1
				x1 = imread('train/' + i)
				x2 = imread('train/' + j)
				n = random.sample(nl,3)
				n1 = imread('noise/' + n[0])
				n2 = imread('noise/' + n[1])
				n3 = imread('noise/' + n[2])
				f = (a*x1 + (1-a)*x2 + b*(n1 + n2 +n3))/2.2
				f = np.asarray(f,dtype = np.uint8)			
				spl.append(spc)			
				nml.append(spc +'_'+str(p))
				imsave('aug_train/' + spc +'_' +  str(p) + '.png',f)
			

df = pd.DataFrame({
	'name':nml,
	'speices':spl
	})
			
df.to_csv('augmented.csv')

		
			
