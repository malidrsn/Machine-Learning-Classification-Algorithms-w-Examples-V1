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


# verilerimizi initialize(initializing parameters) edelim ve sigmoud function
# dimension future sayimiza göre atanır bizim 30 future'miz var o yüzden dimension 30

def initialize_weight_and_bias(dimension):
    w = np.full((dimension, 1), 0.01)  # 30'a 1'lik 0.01'lerden oluşan değer ata demektir.
    b = 0.0
    return w, b


# w, b = initialize_weight_and_bias(30)
# print(w,b)
# sigmoid function f(x) = 1/(1+e^-(x))

def sigmoid(z):
    y_head = 1 / (1 + np.exp(-z))
    return y_head


# print(sigmoid(0)) = 0.5 çıkmalı

# forward backward propagation yapalım ve üstteki değerleri birleştirelim
def forward_backward_propagation(w, b, x_train, y_train):
    # forward propagation
    z = np.dot(w.T, x_train) + b  # her bir future w ile çarpılması 30,1 * 30,455 -> (1,30)*(30,455) matris çarpımı
    y_head = sigmoid(z)
    loss = -y_train * np.log(y_head) - (1 - y_train) * np.log(1 - y_head)
    cost = (np.sum(loss)) / x_train.shape[1]  # örnek sayısını veriyor(shape1 = 455) ve
    # denklem sonucundan scaling yapılmış oluyor

    # backward propagation
    derivative_weight = (np.dot(x_train, ((y_head - y_train).T))) / x_train.shape[1]  # türev and scaling
    derivative_bias = np.sum(y_head - y_train) / x_train.shape[1]
    gradients = {"derivative_weight": derivative_weight, "derivative_bias": derivative_bias}  # dictionary olarak verir
    return cost, gradients  # slope return ediliyor.


# update weight and bias (updating parameters )
def update(w, b, x_train, y_train, learning_rate, number_of_iteration):
    cost_list = []
    cost_list2 = []
    index = []
    # updating(learning) parameters is number_of_iterarion times
    for i in range(number_of_iteration):
        # make forward and backward propagation and find cost and gradients
        cost, gradients = forward_backward_propagation(w, b, x_train, y_train)
        cost_list.append(cost)
        # lets update
        w = w - learning_rate * gradients["derivative_weight"]  # türev
        b = b - learning_rate * gradients["derivative_bias"]  # türev
        if i % 10 == 0:
            cost_list2.append(cost)
            index.append(i)
            print("Cost after iteration %i: %f" % (i, cost))
    # we update(learn) parameters weights and bias
    parameters = {"weight": w, "bias": b}
    plt.plot(index, cost_list2)
    plt.xticks(index, rotation='vertical')
    plt.xlabel("Number of Iterarion")
    plt.ylabel("Cost")
    plt.show()
    return parameters, gradients, cost_list


# parameters, gradients, cost_list = update(w, b, x_train, y_train, learning_rate = 0.009,number_of_iterarion = 200)

# prediction
def predict(w, b, x_test):
    # x_test is a input for forward propagation
    z = sigmoid(np.dot(w.T, x_test) + b)
    Y_prediction = np.zeros((1, x_test.shape[1]))
    # if z is bigger than 0.5, our prediction is sign one (y_head=1),
    # if z is smaller than 0.5, our prediction is sign zero (y_head=0),
    for i in range(z.shape[1]):
        if z[0, i] <= 0.5:
            Y_prediction[0, i] = 0
        else:
            Y_prediction[0, i] = 1

    return Y_prediction


# predict(parameters["weight"],parameters["bias"],x_test)


def logistic_regression(x_train, y_train, x_test, y_test, learning_rate, num_iterations):
    # initialize
    dimension = x_train.shape[0]  # 30 future oludğundan dimension 30
    w, b = initialize_weight_and_bias(dimension)
    # do not change learning rate
    parameters, gradients, cost_list = update(w, b, x_train, y_train, learning_rate, num_iterations)

    y_prediction_test = predict(parameters["weight"], parameters["bias"], x_test)
    # y_prediction_train = predict(parameters["weight"], parameters["bias"], x_train)

    # Print test Errors
    # print("train accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_train - y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))


logistic_regression(x_train, y_train, x_test, y_test, learning_rate=5, num_iterations=150)
