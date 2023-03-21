import numpy as np
import pywt
import pywt.data
from scipy.stats import mode, kurtosis, entropy, skew
import cv2
import pandas as pd
import os
# Import the math function from functions_main.py
from functions_main import entropy_man, kurtosis_man1, skew_man

features_list = []
df_final = pd.DataFrame(features_list)
path = "C:/Users/palla/Desktop/GSI_project/Dexfile_conversion/Benign_grayscale/"
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
        
        coefflist = [CA2, CH2, CV2, CD2]
        mylist = []

        # Stastical Feature extraction: 
        for i in range(len(coefflist)): 
            mylist.append(np.min(coefflist[i])) # minimumvalue
            mylist.append(np.max(coefflist[i])) # maximumvalue
            mylist.append(np.mean(coefflist[i])) # meanvalue
            mylist.append(np.median(coefflist[i])) # medianvalue
            modelist, count = mode(coefflist[i], axis=0)
            mylist.append(modelist[0][0]) # modevalue
            mylist.append(np.var(coefflist[i])) # variance
            mylist.append(np.std(coefflist[i])) # standard_deviation
            mylist.append(kurtosis_man1(coefflist[i])) # kurtosis_man: Manually calculated
            mylist.append(entropy_man(coefflist[i])) # entropy_man: Manually calculated
            mylist.append(np.mean(skew_man(coefflist[i]))) # skewness
            

        features_list = [round(num, 4) for num in mylist]
        features_list = [img_name] + features_list
        features_list = np.reshape(features_list, (1,41))
        df = pd.DataFrame(features_list)
        df_final = df_final.append(df, ignore_index=True)

header1 = ["ImageName", 
                "CA_Minimumvalue", "CA_Maximumvalue", "CA_Meanvalue", "CA_Medianvalue", 
                "CA_Modevalue", "CA_Variance", "CA_STD", "CA_Kurtosis", "CA_Entropy", "CA_Skew",
                "CH2_Minimumvalue", "CH2_Maximumvalue", "CH2_Meanvalue", "CH2_Medianvalue", 
                "CH2_Modevalue", "CH2_Variance", "CH2_STD", "CH2_Kurtosis", "CH2_Entropy", "CH2_Skew",
                "CV2_Minimumvalue", "CV2_Maximumvalue", "CV2_Meanvalue", "CV2_Medianvalue", 
                "CV2_Modevalue", "CV2_Variance", "CV2_STD", "CV2_Kurtosis", "CV2_Entropy", "CV2_Skew",
                "CD2_Minimumvalue", "CD2_Maximumvalue", "CD2_Meanvalue", "CD2_Medianvalue", 
                "CD2_Modevalue", "CD2_Variance", "CD2_STD", "CD2_Kurtosis", "CD2_Entropy", "CD2_Skew"]

df_final.to_excel('my_data.xlsx', index=False, header=header1)
