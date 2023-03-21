import numpy as np
import pywt
import pywt.data
from scipy.stats import mode, kurtosis, entropy, skew
import cv2
import pandas as pd
import os
from functions_main import entropy_man, kurtosis_man, kurtosis_man1, skew_man, fun_skewness

features_list = []
df_final = pd.DataFrame(features_list)
path = "./test_images/"
for entries in os.scandir(path):
    if entries.is_file():
        imgpath = entries.path
        img = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)
        img_name = imgpath.split('/')[-1]


        Coeff = pywt.wavedec2(img, 'haar', level=2)


        # Extracting the coefficients. 
        CA2 = Coeff[0]
        (CH1, CV1, CD1) = Coeff[-1]
        (CH2, CV2, CD2) = Coeff[-2]
        
        coefflist = [CH2]
        mylist = []

        # Stastical Feature extraction: 
        for i in range(len(coefflist)): 
            mylist.append(np.mean(skew_man(coefflist[i])))
            mylist.append((fun_skewness(coefflist[i])))
            mylist.append(np.mean(skew(coefflist[i], axis=0)))
            mylist.append((skew(coefflist[i], axis=None)))
            
        
        features_list = [round(num, 4) for num in mylist]
        features_list = [img_name] + features_list
        features_list = np.reshape(features_list, (1,5))
        df = pd.DataFrame(features_list)
        df_final = df_final.append(df, ignore_index=True) 


df_final.to_excel('test_loop.xlsx', index=False)
# print(mylist)