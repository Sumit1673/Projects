import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
data_df = pd.read_csv("sample_gd.csv")
data_features = np.zeros([10,8])
i=0
for col in data_df.columns:
    data_features[:,i] = data_df[col]
    i+=1


# Create plot
fig = plt.figure()
plt.scatter(data_features[:,0], data_features[:,1], marker='s' ,c='r', alpha=0.5, label='w1')
plt.scatter(data_features[:,2], data_features[:,3], marker='<',c='b', alpha=0.5,label='w2')
# plt.scatter(data_features[:,4], data_features[:,5],marker='>', c='g', alpha=0.5,label='w3')
# plt.scatter(data_features[:,6], data_features[:,7], marker='d',c='c', alpha=0.5, label='w4')

plt.title('4 classes')
plt.xlabel('x1 features')
plt.ylabel('x2 features')
plt.legend()
plt.show()


# print(data_df)
# data_mat = np.zeros([10,8])
# colm =0
# for each_col in range(0,7):
#     data_mat[:, colm] = data_df
#     colm +=1
#     if colm >7:
#         colm=0
# data_mat = np.round(data_mat, 2)
# row = np.size(data_mat,0)
#