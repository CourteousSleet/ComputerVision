import numpy as np
import matplotlib.pyplot as plt
import cv2


def rgb2gs2(_image):
    return np.dot(_image[:, :, :3], [0.299, 0.587, 0.144]).astype(np.uint8)


def box_filter3(_image):
    width, height = _image.shape
    expanded = np.zeros((width + 2, height + 2))
    expanded[1:-1, 1:-1] = _image
    _filtered = np.zeros(_image.shape)
    for i in range(width):
        for j in range(height):
            _filtered[i, j] = 0.11 * (expanded[i, j] + expanded[i, j + 1] + expanded[i, j + 2] +
                                      expanded[i + 1, j] + expanded[i + 1, j + 1] + expanded[i + 1, j + 2] +
                                      expanded[i + 2, j] + expanded[i + 2, j + 1] + expanded[i + 2, j + 2])
    return _filtered


if __name__ == '__main__':
    image = plt.imread('Resources/Psyj.jpg')
    _, ((a1, a2), (b1, b2), (c1, c2), (d1, d2)) = plt.subplots(4, 2)
    a1.imshow(image)
    filtered = box_filter3(rgb2gs2(image))
    a2.imshow(filtered, cmap='gray')

    kernel = np.array([[1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1]]) / 25

    gauss_kernel = np.array([[1, 4, 7, 4, 1],
                             [4, 16, 26, 16, 4],
                             [7, 26, 41, 26, 7],
                             [4, 16, 26, 16, 4],
                             [1, 4, 7, 4, 1]]) / 273

    sobel_x = np.array([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ])

    sobel_y = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])

    filtered2 = cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_REFLECT)
    b1.imshow(filtered2)

    filtered3 = cv2.filter2D(image, -1, gauss_kernel, borderType=cv2.BORDER_REFLECT)
    b2.imshow(filtered3)

    sobel_x_image = cv2.filter2D(rgb2gs2(image), -1, sobel_x, borderType=cv2.BORDER_REFLECT)
    sobel_y_image = cv2.filter2D(rgb2gs2(image), -1, sobel_y, borderType=cv2.BORDER_REFLECT)
    sobel_add = sobel_x_image + sobel_y_image
    c1.imshow(sobel_add, cmap='gray')
    c2.imshow(sobel_x_image, cmap='gray')
    d1.imshow(sobel_y_image, cmap='gray')

    plt.show()
