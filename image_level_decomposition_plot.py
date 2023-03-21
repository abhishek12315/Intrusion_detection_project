'''Plot the images in the wavelet transform'''
import pywt
import matplotlib.pyplot as plt
import cv2

img = cv2.imread("./test_images/grayscale3.png", cv2.IMREAD_GRAYSCALE)
# img = cv2.resize(img, (32,32))


Coeff = pywt.wavedec2(img, 'haar', level=2)


# Extracting the coefficients. 
CA2 = Coeff[0]
(CH1, CV1, CD1) = Coeff[-1]
(CH2, CV2, CD2) = Coeff[-2]

plt.figure(figsize=(20,20))

plt.subplot(2,2,1)
plt.imshow(CA2, cmap=plt.cm.gray)
plt.title("CA2: Approximation Coeff", fontsize=30)

plt.subplot(2,2,2)
plt.imshow(CH2, cmap=plt.cm.gray)
plt.title("CH2: Horizontal Coeff", fontsize=30)

plt.subplot(2,2,3)
plt.imshow(CV2, cmap=plt.cm.gray)
plt.title("CV2: Vertical Coeff", fontsize=30)

plt.subplot(2,2,4)
plt.imshow(CD2, cmap=plt.cm.gray)
plt.title("CD2: Diagonal Coeff", fontsize=30)

arr, Coeff_slices = pywt.coeffs_to_array(Coeff)
plt.figure(figsize=(20,20))
plt.imshow(arr, cmap=plt.cm.gray)
plt.title("All Wavelet Coeff", fontsize=30)

plt.figure(figsize=(20,20))

plt.subplot(2,2,2)
plt.imshow(CH1, cmap=plt.cm.gray)
plt.title("CH1: Horizontal Coeff", fontsize=30)

plt.subplot(2,2,3)
plt.imshow(CV1, cmap=plt.cm.gray)
plt.title("CV1: Vertical Coeff", fontsize=30)

plt.subplot(2,2,4)
plt.imshow(CD1, cmap=plt.cm.gray)
plt.title("CD1: Diagonal Coeff", fontsize=30)


plt.show()
