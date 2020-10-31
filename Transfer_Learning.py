
# coding: utf-8

# In[1]:


from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model 
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import backend as k 
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
from keras.datasets import cifar100
import matplotlib.pyplot as plt
import scipy.ndimage
import numpy as np
from keras.utils import np_utils
import pickle

from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE
from sklearn.manifold import LocallyLinearEmbedding

nb_classes = 100

with open('./train_data', 'rb') as f:
    train_data = pickle.load(f)
    train_labels= pickle.load(f)
with open('./test_data', 'rb') as f:
    test_data = pickle.load(f)
x_train = np.transpose(np.reshape(train_data,(-1,3,32,32)), (0,2,3,1))
x_test = np.transpose(np.reshape(test_data,(-1,3,32,32)), (0,2,3,1))
y_train = np.reshape(train_labels[:1000],(-1,))

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train = x_train/255.
x_test = x_test/255.

y_train =  np_utils.to_categorical(y_train, nb_classes)


img_width, img_height = 256, 256
model = applications.inception_v3.InceptionV3(weights = "imagenet", include_top=False, input_shape = (img_width, img_height, 3))

model.summary()

# yval =  np_utils.to_categorical(yval, nb_classes)
steps = 10

xtrain_features = np.zeros((1,2048))
xtest_features = np.zeros((1,2048))

for i in range(int(x_train.shape[0]/steps)):
    if i%50 == 0:
        print(i)
    xtrain = x_train[steps*(i):steps*(i+1),:,:,:]
    xtrain = scipy.ndimage.zoom(xtrain, (1,8,8,1), order=0)

    # extract the features
    features = model.predict(xtrain)
    features = np.mean(np.reshape(features, (-1,6*6,2048)), axis=1)
    xtrain_features = np.concatenate((xtrain_features, features), axis=0)

for i in range(int(x_test.shape[0]/steps)):
    xtest = x_test[steps*(i):steps*(i+1),:,:,:]
    xtest = scipy.ndimage.zoom(xtest, (1,8,8,1), order=0)

    # extract the features
    features = model.predict(xtest)
    features = np.mean(np.reshape(features, (-1,6*6,2048)), axis=1)
    xtest_features = np.concatenate((xtest_features, features), axis=0)

# ytrain = y_train[steps*(i):steps*(i+1),:]
# xval = x_train[400:500,:,:,:]
# yval = y_train[400:500,:]
#xval = scipy.ndimage.zoom(xval, (1,8,8,1), order=0)

xtrain_features = xtrain_features[1:,:] 
xtest_features = xtest_features[1:,:] 
print(xtrain_features.shape)
print(xtest_features.shape)

np.save('xtrain_features_mean_inceptionv3.npy', xtrain_features)
np.save('xtest_features_mean_inceptionv3.npy', xtest_features)

xtrain_features = np.load('xtrain_features_mean_inceptionv3.npy')
xtest_features = np.load('xtest_features_mean_inceptionv3.npy')

train_feat = xtrain_features
#val_feat = xtrain_features[45000:,:]
test_feat = xtest_features

_ , (test , y_test) = cifar100.load_data()


n_components_list = [550, 750, 1024]
C_list = [1, 10, 100]
for n_components in n_components_list:
    for C in C_list:
        print(n_components)
        print(C)
        #Dimensionality Reduction on Features
        pca = PCA(n_components=n_components)
        components_pca = pca.fit(train_feat)
        X_train= pca.transform(train_feat)
        X_test= pca.transform(test_feat)

        #Classifier SVM Linear Kernel 
        clf = SVC(C=C, gamma='auto', kernel='rbf')
        clf = clf.fit(X_train,train_labels)
        predictions_tr = (clf.predict(X_test))

        #20% Test Data Accuracy
        test_acc = accuracy_score(y_test,predictions_tr)
        print("Test Accuracy: %0.4f" % test_acc)


# LDA Classifier


n_components_list = [1024]
C_list = [1]
for n_components in n_components_list:
    for C in C_list:
        print(n_components)
        #Dimensionality Reduction on Features
        pca = PCA(n_components=n_components)
        components_pca = pca.fit(train_feat)
        X_train= pca.transform(train_feat)
        X_test= pca.transform(test_feat)
        
        #Classifier SVM Linear Kernel
        clf = LinearDiscriminantAnalysis()
        clf.fit(X_train, train_labels)
        predictions_tr = (clf.predict(X_test))

        #20% Test Data Accuracy
        test_acc = accuracy_score(y_test,predictions_tr)
        print("Test Accuracy: %0.4f" % test_acc)
        
# TSNE  Classifier    

n_components_list = [3]
C_list = [1]
for n_components in n_components_list:
    for C in C_list:
        print(n_components)
        print(C)
        X_train_Embedded_tsne = TSNE(n_components=n_components).fit_transform(train_feat)
        X_test_Embedded_tsne = TSNE(n_components=n_components).fit_transform(test_feat)
        
        #Classifier SVM Linear Kernel 
        clf = SVC(C=C, gamma='auto', kernel='rbf')
        clf = clf.fit(X_train_Embedded,train_labels)
        predictions_tr = (clf.predict(X_test_Embedded))

        #20% Test Data Accuracy
        test_acc = accuracy_score(y_test,predictions_tr)
        print("Test Accuracy: %0.4f" % test_acc)
        
        
# LLE Classifier
n_components_list = [2, 10, 50, 100]
C_list = [1]
for n_components in n_components_list:
    for C in C_list:
        print(n_components)
        print(C)
        X_train_Embedded_LLE = LocallyLinearEmbedding(n_components=n_components).fit_transform(train_feat)
        X_test_Embedded_LLE = LocallyLinearEmbedding(n_components=n_components).fit_transform(test_feat)
        
        print(X_train_Embedded_LLE.shape)
        #Classifier SVM Linear Kernel 
        clf = SVC(C=C, gamma='auto', kernel='rbf')
        clf = clf.fit(X_train_Embedded_LLE,train_labels)
        predictions_tr = (clf.predict(X_test_Embedded_LLE))

        #20% Test Data Accuracy
        test_acc = accuracy_score(y_test,predictions_tr)
        print("Test Accuracy: %0.4f" % test_acc)




