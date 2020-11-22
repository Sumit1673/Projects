import numpy as np


def non_maximal_suppression(grad_img, grad_angles):
    """
    :param grad_img_mat: matrix of gradient obtained using this method
    https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_canny/py_canny.html
    :param grad_angles_mat:  """
    w, h = grad_img.shape
    grad_angles = grad_angles*180/np.pi

    # keeping all the angles under a range of 0 to 180
    grad_angles[grad_angles < 0] += 180
    suppressed_img = np.zeros(shape=(w, h))
    for i in range(0, w):
        for j in range(0, h):
            try:
                q = 255
                r = 255

                # angle 0
                if (0 <= grad_angles[i, j] < 22.5) or (157.5 <= grad_angles[i, j] <= 180):
                    q = grad_img[i, j + 1]
                    r = grad_img[i, j - 1]

                # angle 45
                elif 22.5 <= grad_angles[i, j] < 67.5:
                    q = grad_img[i + 1, j - 1]
                    r = grad_img[i - 1, j + 1]
                # angle 90
                elif 67.5 <= grad_angles[i, j] < 112.5:
                    q = grad_img[i + 1, j]
                    r = grad_img[i - 1, j]
                # angle 135
                elif 112.5 <= grad_angles[i, j] < 157.5:
                    q = grad_img[i - 1, j - 1]
                    r = grad_img[i + 1, j + 1]

                if grad_img[i, j] >= q and grad_img[i, j] >= r:
                    suppressed_img[i, j] = grad_img[i, j]
                else:
                    suppressed_img[i, j] = 0

            except IndexError as e:
                pass

    return suppressed_img
