import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# veri yükleme
data = pd.read_csv("data.csv")

# gereksizleri kaldırma
data.drop(["Unnamed: 32", "id"], axis=1, inplace=True)

# diagnosis kısmını object'ten integer'e çevirmek
data.diagnosis = [1 if each == "M" else 0 for each in data.diagnosis]

# verileri atama
y = data.diagnosis.values
x_data = data.drop(["diagnosis"], axis=1)

# normalization
x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data))

# train test split
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# KNN algorithm
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=4)  # n_neighbors = k değeri default 5'tir.
knn.fit(x_train, y_train)
prediction = knn.predict(x_test)
print(prediction)
print(y_test)
print("KNN k={} score {}:".format(4, knn.score(x_test, y_test)))

# find k value
score_list = []
for each in range(1, 15):
    knn2 = KNeighborsClassifier(n_neighbors=each)
    knn2.fit(x_train, y_train)
    score_list.append(knn2.score(x_test, y_test))

plt.plot(range(1, 15), score_list)
plt.xlabel("k values")
plt.ylabel("accuracy")
plt.show()
# bakınca k =4 için en doğru çıkıyor
