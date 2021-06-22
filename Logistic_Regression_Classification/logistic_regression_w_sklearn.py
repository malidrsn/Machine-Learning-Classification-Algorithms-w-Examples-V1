# "datasetimiz iyi veya kötü huylu tümörlerimiz" M kötü B iyi huylu tümör demek
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

# normalization ölçekleme (x-min(x))/(max(x)- min(x))
x = ((x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data))).values

# Train Test Split
# data -> logistic regression -> Model
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
# print(x_train)
# print(x_test)
# print(y_train)
# print(y_test)

# satır ve sütunların yerlerini değiştirelim ki future'lar columnda olsun 455x30 -> 30 future 455 sample 30*455 olucak
x_train = x_train.T
x_test = x_test.T
y_train = y_train.T
y_test = y_test.T

print("x_train : ", x_train.shape)
print("x_test : ", x_test.shape)
print("y_train : ", y_train.shape)
print("y_test : ", y_test.shape)

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(x_train.T, y_train.T)  # 30*455 -> 455*30

print("Test Accuracy {}".format(lr.score(x_test.T, y_test.T)*100))
