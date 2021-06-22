import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv("data.csv")
# print(data.head())  # baştan 5 satır getir
# print(data.tail())  # sondan 5 satır getir

data.drop(["Unnamed: 32", "id"], axis=1, inplace=True)  # kullanılmayacak colonları remove et
M = data[data.diagnosis == "M"]
B = data[data.diagnosis == "B"]
# M.info() kötü huylu tümör
# B.info() iyi huylu tümör 357 tane

plt.scatter(M.radius_mean, M.texture_mean, color="red", label="Kötü", alpha=0.3)  # alpha=saydamlik
plt.scatter(B.radius_mean, B.texture_mean, color="blue", label="İyi")
plt.xlabel("radius_mean")
plt.ylabel("texture_mean")
plt.legend()
plt.show()

# KNN = k nearest neighbour = K en yakın komşu algoritması aşamaları
# k = yakın komşu sayısı se.
# en yakın data noktaları bul
# k en yakın komşu arasında hangi class'tan kaç tane var hesapla
# eklediğimiz test ettiğimiz point yada class'a ait tespit et
# öklid manhattan ve başka mesafe uzaklıklarına göre hesaplanır k komşu uzaklıkları en yaygın öklid
# normalization yapmak önemlidir.
