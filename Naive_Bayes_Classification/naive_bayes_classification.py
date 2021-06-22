# olasılık üzerine çalışan bir sınıflandırma yöntemidir.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("data.csv")

data.drop(["Unnamed: 32", "id"], axis=1, inplace=True)  # axis1 = column inplace = drop et ve kaydet demek
# print(data.info())

# diagnosis objec olduğu için kategorik veya float yapmam lazım
# diagnosis class'ını int türüne çeviriyoruz
data.diagnosis = [1 if each == "M" else 0 for each in data.diagnosis]
# print(data.info())

y = data.diagnosis.values
x_data = data.drop(["diagnosis"], axis=1)

# normalization
x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data))

# train test split
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Naive Bayes Algorithm
from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()
nb.fit(x_train, y_train)

# test
print("Accuracy of Naive Bayes Algorithm :", nb.score(x_test, y_test))
