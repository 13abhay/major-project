from keras.layers import Conv2D,Reshape,MaxPooling2D
from keras.models import Sequential
from keras.layers import InputLayer
from keras.layers import Dense,Flatten
from keras.optimizers import Adam
from keras.layers import Dropout
#import matplotlib.pyplot as plt

import numpy as np
from sklearn.cross_validation import train_test_split
#from scipy import misc
#from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import load_img,array_to_img,img_to_array
import pandas as pd
#import seaborn as sns
import gc

meta_train = pd.read_csv('aug_train/augmented.csv')
X,y  = [],[]

for i in meta_train.name:
    X.append(img_to_array(load_img('aug_train/' + i + '.png')).ravel())
    

X = np.array(X)/255
y = pd.get_dummies(meta_train.speices)

train_x,test_x,train_y,test_y =  train_test_split(X,y,test_size = .1)
print(X.shape)
print(y.shape)
print(train_x.shape)
print(train_y.shape)
print(test_x.shape)
print(test_y.shape)
del X
del y
gc.collect()

img_size = 48
img_size_flat = 129*295*3
img_shape = [129,295,3]
img_shape_full = [129,295,3]
num_classes = 82

def Model():
    model = Sequential()
    model.add(InputLayer(input_shape=(img_size_flat,)))
    #print("shape outputted by the Input layer: ",model.output_shape)
    model.add(Reshape(target_shape = img_shape_full))
    #print("shape outputted by the reshape layer: ",model.output_shape)
    model.add(Conv2D(filters = 64,kernel_size = 5,input_shape = img_shape_full,activation = 'relu',strides = (1,1),padding = 'same',name = 'layer_conv1'))
    #print("shape outputted by the first convolutional layer: ",model.output_shape)
    model.add(MaxPooling2D(pool_size = (3,3),strides = (2,2)))
    #print("shape outputted by the MaxPooling Layer layer: ",model.output_shape)
    model.add(Conv2D(filters = 64,kernel_size = 5,activation = 'relu',strides = (2,2),padding = 'same',name = 'layer_conv2'))
    #print("shape outputted by the second Convolutional layer: ",model.output_shape)
    model.add(MaxPooling2D(pool_size = (3,3),strides = (2,2)))
    #print("shape outputted by the second Maxpooling layer: ",model.output_shape)
    model.add(Conv2D(filters = 128,kernel_size = 4,activation = 'relu'))
    #print("shape outputted by the convolutional layer: ",model.output_shape)
    model.add(MaxPooling2D(pool_size = (3,3), strides = (2,2)))


    model.add(Dropout(rate = .3))
    #print("shape outputted by the Dropout layer: ",model.output_shape)
    model.add(Flatten())
    #print("shape outputted by the after flatten layer: ",model.output_shape)
    model.add(Dense(4096,activation = 'relu'))
    #print("shape outputted by the after Dense layer: ",model.output_shape)
    model.add(Dense(num_classes,activation = 'softmax'))
    optimizer = Adam(.001)
    model.compile(optimizer = optimizer,loss = 'categorical_crossentropy', metrics = ['accuracy'])
    model.fit(x= train_x,y=train_y,epochs = 10,batch_size = 16,validation_split = .1)
    result = model.evaluate(x = test_x,y = test_y)
    for name,value in zip(model.metrics_names,result):
        print('{0}: {1}'.format(name,value))
    return model

m = Model()

#model.add(Regression(optimizer = 'momentum',loss = 'categoriacal_crossentropy'))   

#m.save('Models/fer_cnn_model_1.h5')

'''
m.save_weights('Models/fer_cnn_model_2_weights.h5')
with open('Models/fer_cnn_model_2_json.json','w') as json_file:
    json_file.write(m.to_json())
'''
