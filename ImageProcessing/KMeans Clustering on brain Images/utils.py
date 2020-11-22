from scipy.io import loadmat
import matplotlib.pyplot as plt

annots = lambda mat_file: loadmat(mat_file)

def extract_imgs_from_mat(mat_file='T1_T2_PD.mat'):
    t1_t2_d_annots = annots(mat_file)
    t1_image = t1_t2_d_annots['t1']
    t2_image = t1_t2_d_annots['t2']
    pd_image = t1_t2_d_annots['pd']
    plt.subplot(2,2,1), plt.imshow(t1_image, cmap='gray')
    plt.subplot(2,2,2), plt.imshow(t2_image, cmap='gray')
    plt.subplot(2,2,3), plt.imshow(pd_image, cmap='gray')
    plt.show()

    # plt.imsave('t1.png', t1_image)
    # plt.imsave('t2.png', t2_image)
    # plt.imsave('pd.png', pd_image)
    return (t1_image, t2_image, pd_image)
extract_imgs_from_mat()