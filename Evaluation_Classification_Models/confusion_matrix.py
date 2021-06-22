# TN FP
# FN TP


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

# random forest algorithm
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, random_state=1)
rf.fit(x_train, y_train)

print("random forest accuracy :", rf.score(x_test, y_test))

# confusion matrix
y_pred = rf.predict(x_test)
y_true = y_test

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_true, y_pred)
print(cm)

# confusion  matrix visualization
import seaborn as sns
import matplotlib.pyplot as plt

f, ax = plt.subplots(figsize=(5, 5))
sns.heatmap(cm, annot=True, linewidths=0.5, linecolor="red", fmt=".0f", ax=ax)
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.show()
