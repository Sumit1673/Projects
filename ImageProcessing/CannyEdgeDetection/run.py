from canny_edge import EdgeFilter
from non_maximal_supression import non_maximal_suppression as nms
from hysteresis_thrshold import HysterisisThreshold
import subprocess
import sys

import matplotlib.pyplot as plt

try:
    from scipy.io import loadmat
except ImportError as e:
    print(e, "Scipy Missing")
    subprocess.check_call([sys.executable, "-m", "pip", "install", 'scipy'])
    from scipy.io import loadmat
    
annots = lambda mat_file: loadmat(mat_file)


brain_annots = annots('BrainWeb.mat')
img_array = brain_annots['I']
plt.subplot(2,2,1), plt.imshow(img_array, cmap='gray'), plt.title("Input Image")
# print(type(img))
c = EdgeFilter(img_array, 1)
grads, orient = c.run()
plt.subplot(2,2,2), plt.imshow(grads, cmap='gray'), plt.title("Gradient Image")
smooth_img = nms(grads, orient)
plt.subplot(2,2,3), plt.imshow(smooth_img, cmap='gray'), plt.title("NMS Image")
thr = HysterisisThreshold(smooth_img, low_thr=0.02, high_thr=0.09)
final_img = thr.run()
plt.subplot(2,2,4), plt.imshow(final_img, cmap='gray'), plt.title("Final Image")
plt.show()
