# ensamble learning modelidir. birden fazla algoritmanın birleşmesi ile oluşan model. 100x Decision Tree = Random Forest gibi.
# daha dayanıklı ve accuracy yüksek çıkıyor
# n tane sample seç ve bu seçilen sample'ler sub sample olarak adlandırılır.
# daha sonra dt ile train edilir.
# tekrar tekrar n tane dt kullanılır ve bunlar toplanır.çıkan sonuca göre ortalama bir sonuç çıkar

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
