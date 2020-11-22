import numpy as np
import matplotlib.pyplot as plt
STRONG = 255
WEAK = 50


class HysterisisThreshold:
    def __init__(self, nms_img, low_thr=0.2, high_thr=0.9):
        self.h_thr = np.max(nms_img)*high_thr
        self.l_thr = self.h_thr*low_thr
        self.inp_img = np.array(nms_img)
        self.binary_img = np.zeros(shape=self.inp_img.shape)

    def run(self, weak=25, strong=255):
        self.threshold(weak, strong)
        return self.hysterisis(weak, strong)

    def threshold(self, weak=25, strong=255):
        """
        thresholding above the high threshold value to 255
        thresholding below the low threshold value to 0
        thresholding between high threshold and the low threshold value to weak
        :return:
        """
        strong_row, strong_col = np.where(self.inp_img >= self.h_thr)
        weak_row, weak_col = np.where((self.inp_img <= self.h_thr) & (self.inp_img >= self.l_thr))
        zeros_row, zeros_col = np.where(self.inp_img < self.l_thr)

        self.binary_img[strong_row, strong_col] = strong
        self.binary_img[weak_row, weak_col] = weak
        self.binary_img[zeros_row, zeros_col] = 0

    def hysterisis(self, weak=25, strong=255):
        x,y = self.binary_img.shape
        out_img = np.zeros(shape=(x,y))
        for i in range(1, x-1):
            for j in range(1, y-1):
                if self.binary_img[i][j] == weak:
                    try:
                        if ((self.binary_img[i + 1, j - 1] == strong) or (self.binary_img[i + 1, j] == strong) or (self.binary_img[i + 1, j + 1] == strong)
                                or (self.binary_img[i, j - 1] == strong) or (self.binary_img[i, j + 1] == strong)
                                or (self.binary_img[i - 1, j - 1] == strong) or (self.binary_img[i - 1, j] == strong) or (
                                        self.binary_img[i - 1, j + 1] == strong)):
                            self.binary_img[i, j] = strong
                        else:
                            self.binary_img[i, j] = 0
                    except:
                        pass
        return self.binary_img


if __name__== "__main__":
    binary_img= plt.imread('brain_web.png')
    # print(0.02*np.max(img)*0.09)
    th = HysterisisThreshold(binary_img)
    i = th.run()
    plt.imshow(i, cmap ='gray')
    plt.show()

