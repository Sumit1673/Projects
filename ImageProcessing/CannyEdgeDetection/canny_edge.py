from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt



class EdgeFilter:
    def __init__(self, img, sigma):
        self.sigma = sigma
        self.img= img
        self.img_shape = img.shape

    def get_filter(self, f_shape=(3, 3)):

        m, n = [(ss - 1.) / 2. for ss in f_shape]
        y, x = np.ogrid[-m:m + 1, -n:n + 1]
        h = np.exp(-(x * x + y * y) / (2. * self.sigma * self.sigma))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        sumh = h.sum()
        if sumh != 0: # for noise removal, we need low pass filter. so the sum should not be zero
            h /= sumh
        return h

    def pad_img(self, img, f_size, stride=1):
        """
        padding the inp with zeros.
        P = (F_size - Stride)/2
        :param f_size:
        :param img:
        :param stride:
        :return:
        """
        pad = int((f_size[0] - stride)/2)

        padded_img = np.pad(img, ((pad, pad), (pad, pad)), mode='constant', constant_values=(0, 0))
        return padded_img, pad

    def smoothening(self, kernel, s=1):
        if kernel is None:
            kernel = self.get_filter()
        k_shape = kernel.shape

        padded_img, pad = self.pad_img(self.img, k_shape, s)

        # Convolutional output size
        conv_x = int(((self.img_shape[0] - k_shape[0] + 2 * pad) / s) + 1)
        con_y = int(((self.img_shape[1] - k_shape[1] + 2 * pad) / s) + 1)
        output = np.zeros((conv_x, con_y))

        output = self.convolution(self.img, output, padded_img, kernel, s)

        return output

    @staticmethod
    def convolution(org_img, out_img, padded_img, kernel, s):
        org_img_shape_x, org_img_shape_y = org_img.shape[0], org_img.shape[1]
        k_shape = kernel.shape
        for v_shift in range(org_img_shape_y):
            # if v_shift > self.img_shape[1] - k_shape[1]:
            #     break
            # if v_shift % s == 0:
            v_shift = v_shift*s
            for h_shift in range(org_img_shape_x):
                # if h_shift > self.img_shape[0] - k_shape[0]:
                #     break
                # try:
                #     if h_shift % s == 0:
                h_shift = h_shift*s
                out_img[h_shift, v_shift] = np.multiply(
                    padded_img[h_shift:h_shift + k_shape[0], v_shift:v_shift + k_shape[1]],
                    kernel).sum()
                # except:
                #     break
        return out_img

    def run(self):
        kernel = self.get_filter((5,5))
        smooth_img = self.smoothening(kernel)
        return self.intensity_gradient(smooth_img)

    def intensity_gradient(self, smoothened_img):
        """
        To obtain the intensity gradient (derivative) smoothened image filtered with a Sobel kernel in x and y direction.
        These gradients in x and y are seperate images. Over these two images, we can find edge gradient
        and direction for each pixel as follows:

        Edge_Gradient(G) = sqrt{G_x^2 + G_y^2}

        Angle(theta) = tan^(-1) (G_y/G_x)

        Gradient direction is always perpendicular to edges. It is rounded to one of four angles representing vertical,
        horizontal and two diagonal directions.
        :return:
        """
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        smooth_img_x_shape, smooth_img_y_shape = smoothened_img.shape[0], smoothened_img.shape[1]
        pad_img, pad = self.pad_img(smoothened_img, (3,3))
        s = 1


        # Get G_x
        # Convolutional output size
        conv_x = int(((smooth_img_x_shape - sobel_x.shape[0] + 2 * pad) / s) + 1)
        con_y = int(((smooth_img_y_shape - sobel_x.shape[1] + 2 * pad) / s) + 1)
        out_img = np.zeros((conv_x, con_y))

        G_x = self.convolution(smoothened_img, out_img, pad_img, sobel_x, s)

        # Get G_y
        G_y = self.convolution(smoothened_img, out_img, pad_img, sobel_y, s)

        edge_gradient = np.sqrt((G_x*G_x + G_y*G_y))

        orientation = np.arctan2(G_y, G_x) # use when angle needed in degree* 180 / np.pi
        # plt.imsave('brain_edge_img.png', edge_gradient)
        # plt.imshow(edge_gradient, cmap='gray')
        # plt.show()
        return (edge_gradient, orientation)


# c = EdgeFilter(img_array, 1)
# print(c.run()[1])
#

