from scipy.stats import kurtosis, entropy, skew
import numpy as np
import pandas as pd
from functions_main import skew_man, entropy_man, kurtosis_man, kurtosis_man1, fun_skewness
import pywt
import cv2

img = cv2.imread("./test_images/grayscale1.png", cv2.IMREAD_GRAYSCALE)
# img = cv2.resize(img, (32,32))


Coeff = pywt.wavedec2(img, 'haar', level=2)


# Extracting the coefficients. 
CA2 = Coeff[0]
(CH1, CV1, CD1) = Coeff[-1]
(CH2, CV2, CD2) = Coeff[-2]


print(np.mean(skew_man(CH2)))
print(np.mean(skew(CH2, axis=0)))
print(skew(CH2, axis=None))
print(fun_skewness(CH2))
# matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])



# create a 3x3 matrix
matrix1 = [13,14,15]
matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10,11,12]])

matrix = np.append(matrix, [matrix1], axis=0)
# calculate the kurtosis of the matrix
matrix_kurtosis = kurtosis_man(matrix)
matrix_kurtosis1 = kurtosis_man1(matrix)

# # create a pandas dataframe from the list
# df = pd.DataFrame(matrix)
# header = ["Column1", "Column2", "Column3"]
# # write the dataframe to an Excel file
# df.to_excel('my_data.xlsx', index=False, header=header)



