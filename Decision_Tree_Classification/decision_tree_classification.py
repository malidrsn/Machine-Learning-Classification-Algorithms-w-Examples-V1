# CART = Classification and regression trees
# classification için splitler ayrılıyor. minimize entropy
# splitler en iyi ayıracak şekilde ayrılıyor.
# birden fazla split kullanılabiliyor
# bölgeleri kendilerine ait noktalara ayırıyoruz yani o bolgede o renkler var gibi

import pandas as pd
import numpy as np

data = pd.read_csv("data.csv")

data.drop(["Unnamed: 32", "id"], axis=1, inplace=True)

# data.info()
# diagnosis object'ten oluştuğundan dolayı onu int yapalım yada kategorical
data.diagnosis = [1 if each == "M" else 0 for each in data.diagnosis]

# scaling (normalize edelim )
y = data.diagnosis.values
x_data = data.drop(["diagnosis"], axis=1)
x = ((x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data)))

# M = kötü huylu tümör
# B = iyi huylu tümör

# train test split
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=42)

# decision tree classification
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier()
dt.fit(x_train, y_train)

print("score : ", dt.score(x_test, y_test))
