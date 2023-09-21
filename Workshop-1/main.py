import numpy as np
import matplotlib.pyplot as plt


def rgb2gs(_image):
    return np.dot(_image[:,:,:3], [0.299, 0.587, 0.144])


def gamma(_image, c, g):
    return (c*(_image**g)).astype(np.uint8)


if __name__ == '__main__':
    #image = plt.imread('./Resources/Psyj.jpg')
    #print(image.shape)
    #gs_image = rgb2gs(image)
    #_, ((a1, a2), (b1, b2)) = plt.subplots(2, 2)
    #a1.imshow(image)
    #a2.hist(image.ravel(), bins=256)
#
    ## Proof that histogram computes for a gray colour.
    #b1.imshow(gs_image, cmap='gray')
    #b2.hist(image.ravel(), bins=256)
    #plt.show()

    _, ((a1, a2), (b1, b2)) = plt.subplots(2, 2)
    image2 = plt.imread('./Resources/Dark.jpg')
    corrected_image2 = gamma(image2, 15, 0.5)
    a1.imshow(image2)
    a2.hist(image2.ravel(), bins=256)

    b1.imshow(corrected_image2)
    b2.hist(corrected_image2.ravel(), bins=256)
    plt.show()
