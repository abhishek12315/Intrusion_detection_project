from scipy.stats import kurtosis, skew
import numpy as np

def entropy_man(matrix):
    _, counts = np.unique(matrix, return_counts=True)
    probabilities = counts / counts.sum()
    entropy = - (probabilities * np.log2(probabilities)).sum()
    return entropy

""" Kurtosis_man and kurtosis_man1 produces the same result. """
def kurtosis_man(matrix):
    flatten_matrix = matrix.flatten()
    kur = kurtosis(flatten_matrix)
    return kur

def kurtosis_man1(matrix):
    mmin = np.mean(matrix)
    m_std = np.std(matrix)
    n = matrix.size
    kur1 = ((1/n) * np.sum(((matrix - mmin) / m_std) ** 4)) - 3
    return kur1

def skew_man(mat): 
    mean = np.mean(mat)
    std = np.std(mat)

    # Calculate the skewness of the matrix manually
    n = mat.size
    skewness = (1/n) * sum(((mat - mean) / std)**3)
    skewness1 = skew(mat, axis=None)
    return skewness

def fun_skewness(arr):
    mean_ = np.mean(arr)
    median_ = np.median(arr)
    std_ = np.std(arr)
    skewness = 3*(mean_ - median_) / std_
    return skewness


# Skewness
""" Skewness of a matrix is a measure of the asymmetry of the distribution of 
the values in the matrix. It indicates whether the values are concentrated on 
one side of the mean or are evenly distributed around the mean.
In a normally distributed matrix, the skewness is zero. A positive skewness 
value indicates that the tail of the distribution is longer on the positive 
side of the mean, while a negative skewness value indicates that the tail of 
the distribution is longer on the negative side of the mean.
Skewness can be calculated using various statistical methods, including the 
Pearson's skewness coefficient, the Bowley's skewness coefficient, and the 
Fisher-Pearson standardized moment coefficient. The most commonly used method 
is Pearson's skewness coefficient, which is defined as the third standardized 
moment of the distribution divided by the cube of the standard deviation"""

# Kurtosis
"""Kurtosis of a matrix is a measure of the tailedness or peakedness of the 
distribution of the values in the matrix. It indicates whether the distribution 
has more or fewer outliers than a normal distribution.
In a normally distributed matrix, the kurtosis is zero. A positive kurtosis 
value indicates that the distribution has heavier tails and a more peaked 
center than a normal distribution, while a negative kurtosis value indicates 
that the distribution has lighter tails and a flatter center than a normal 
distribution.
Kurtosis can be calculated using various statistical methods, including the 
Pearson's kurtosis coefficient, the Bowley's kurtosis coefficient, and the 
Fisher-Pearson standardized moment coefficient. The most commonly used method 
is Pearson's kurtosis coefficient, which is defined as the fourth standardized
moment of the distribution divided by the square of the standard deviation."""

# Entropy: 
"""The entropy of a matrix is a measure of the randomness or unpredictability 
of the values in the matrix. It is a concept from information theory that 
measures the amount of uncertainty or information content in a random variable.
In a matrix, the entropy is calculated as the sum of the product of the 
probability of each value and the logarithm of the probability of each value. 
The entropy is maximum when all values in the matrix are equally likely, and 
minimum when all values in the matrix are the same.
The entropy of a matrix is a useful metric in image processing, pattern 
recognition, and machine learning, as it provides a measure of the complexity 
or diversity of the values in the matrix."""