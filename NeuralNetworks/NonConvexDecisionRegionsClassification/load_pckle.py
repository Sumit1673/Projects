import pickle as pkl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
cols = ['x1','x2','Y']

df_csv = pd.read_csv("data_non_convex.csv", usecols=cols)
training_data = df_csv.values.tolist()
# train_feature = np.zeros(shape=(len(training_data), 2))
# labels = np.zeros(shape=(len(training_data), 1))
x1,x2,l = [], [] , []
train_feature = {'x1':None, 'x2':None,'Y':None}
for i in range(0, len(training_data)):
    x = training_data[i]
    x1.append(x[0])
    x2.append(x[1])
    l.append(x[2])
train_feature= {'x1':x1, 'x2':x2,'Y':l}
pckl_file = open('non_convex_data.pkl', 'wb')
pkl.dump(train_feature,pckl_file)
pckl_file.close()
pkl_file = pd.read_pickle("non_convex_data.pkl")
#
# len_of_pkl_file = len(pkl_file['x1'])
# train_feature = np.zeros(shape=(len_of_pkl_file, 2))
# labels = np.zeros(shape=(len_of_pkl_file, 1))
# x1 = np.asarray(pkl_file['x1'], dtype=np.float32)
# x2 = np.asarray(pkl_file['x2'], dtype=np.float32)
# labels = np.asarray(pkl_file['Y'])
# for i in range(0, len_of_pkl_file):
#     train_feature[i][0] = x1[i]
#     train_feature[i][1] = x2[i]
# plt.scatter(train_feature[:, 0], train_feature[:, 1], c=labels.ravel(), cmap=plt.cm.coolwarm
#             , label="Training Data")
# plt.show()