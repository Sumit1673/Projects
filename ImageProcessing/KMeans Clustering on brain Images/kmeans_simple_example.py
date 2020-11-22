import sys
sys.path.append('../')
import matplotlib.pyplot as plt
import cv2
import numpy as np
import numpy.linalg as LA
import random
from config import get_config
colors = {0:(0,0,255), 1:(0,255,0), 2:(192,0,120),
          4:(128, 128, 0), 3:(255,0,255), 7:(238,130,238), 6:(0,255,0), 5:(129, 0, 0),
          8:(210,105,30), 9:(255,140,0)}

# plt.show
class KMeans:
    def __init__(self, image, norm='L2', K=5, iter_=5):
        self.__image = image
        self.__image = self.__image[:,:, 0]
        if len(self.__image.shape) > 2:
            print('Use a gray scale image')
            return

        self.im_w = self.__image.shape[0]
        self.im_h = self.__image.shape[1]
        self.iter = iter_
        self.__norm = norm
        self.K = K
        self.loss = [-99]
        self.check_freq = 3 # after every 3 iterations



    def kmeans(self):

        centroids = self.select_centroid()
        iters = 0
        labels_dict = dict()
        old_centroid = None
        while not self.converge(old_centroid, centroids, iters):
            old_centroid = centroids
            points_with_labels = self.get_labels(labels_dict, centroids)
            centroids = self.update_centroids(points_with_labels, centroids)
            print('Loss after {} iteration = {}'.format(iters, self.loss[iters]))
            iters += 1
        self.segmentation(points_with_labels)
        return centroids, points_with_labels


    def get_labels(self, labels_dict, centroids):
        """ keys of label_dict are the index of the actual centroid list
            0 means centroid at the index 0 of centroid list. All the
            samples assigned to key 0 are near to centroid at index 0.
        """
        for x in range(self.im_w):
            for y in range(self.im_h):
                dist = []
                for cent in centroids:
                    d = LA.norm(cent-np.array([x,y]))
                    dist.append(d)
                min_ = dist.index(min(dist))
                if min_ not in labels_dict:
                    labels_dict[min_] = [np.array([x,y])]
                else:
                    labels_dict[min_].append(np.array([x,y]))
        return labels_dict

    def update_centroids(self, data_with_labels, centroids):
        # update the centroid position by taking the mean of all the points belonging to each labels
        new_centr = {}
        centroids = []
        for i, v in data_with_labels.items():
            cent = np.mean(v, axis=0)
            # new centroid values are updated. the key value of the dict
            # is the position of the centroid in the centroid list
            # centroid[index] is equal to data_with_labels[key]
            centroids.insert(i, cent)

        return centroids

    def converge(self, old_centroid, new_centroid, iter_):
        # 1st condition
        if iter_ >= self.iter:
            return True
        # 2nd condition no change in loss
        if old_centroid and old_centroid is not None:
            diff = []
            for i in range(len(new_centroid)):
                diff.append(LA.norm(old_centroid[i] - new_centroid[i]))
            self.loss.append(sum(diff)/len(diff))
            if iter_%3 ==0:
                change = self.loss[iter_ -1]/self.loss[iter_-2]*100
                self.check_freq-=1
                if change < 0.05 and self.check_freq==0:
                    return True
        return False

    def select_centroid(self):
        c = []
        for k in range(self.K):
            c.append(np.array([random.randrange(0, self.im_w),(random.randrange(0, self.im_h))]))
        return c

    def segmentation(self, data_labels):
        import copy
        alpha = 0.002
        # rgb_image = cv2.cvtColor(self.__image,cv2.COLOR_GRAY2RGB)
        plt.subplot(1, 2, 1), plt.imshow(self.__image)
        overlay_image = copy.deepcopy(self.__image)
        output = copy.deepcopy(self.__image)
        for i, v in data_labels.items():
            col = colors[i]
            for i in v:
                overlay_image[i[0], i[1], 0] = col[0]
                overlay_image[i[0], i[1], 1] = col[1]
                overlay_image[i[0], i[1], 2] = col[2]

        cv2.addWeighted(overlay_image, alpha, output, 1 - alpha, 0, output)
        plt.subplot(1,2,2), plt.imshow(output)
        plt.show()



if __name__ == '__main__':
    config, _ = get_config()
    img_t1 = plt.imread('t1.png')[:,:, 0]
    img_t2 = plt.imread('t2.png')[:,:, 0]
    img_pd = plt.imread('pd.png')[:,:, 0]
    org_img = np.zeros(shape=(img_t1.shape[0], img_t1.shape[1] ,3))
    org_img[:,:, 1] = img_t1
    org_img[:, :, 1] = img_t2
    org_img[:, :, 1] = img_pd
    kmeans_obj = KMeans(image=org_img, norm=config.norm, K=config.k, iter_=5)
    centroid, data = kmeans_obj.kmeans()
#     for k in range(centroid.shape[0]):
#         dist = [LA.norm(centroid[k]- i_data) for i_data in data]
#         print(dist)
#         new_points_index = (dist.index(min(dist)))
#         new_centroid[k] = data[new_points_index]
#     return new_centroid
#
# print(k_means(centroid=centroid, data=data))
# print([LA.norm(data - centroid[1]) for data in data])


# d2 = find_distance(centroid=centroid[1] ,data=data)
# print(d1)