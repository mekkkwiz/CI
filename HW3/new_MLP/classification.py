from typing import Counter
import numpy as np
import matplotlib.pyplot as plt
from network import Network
from fc_layer import FCLayer
from activation_layer import ActivationLayer
from activations import tanh, tanh_prime
from losses import mse, mse_prime
from fillspace import nomalized_input,Output



# training data
x_train = np.array([[i] for i in nomalized_input])
# print (len(x_train[0][0]))
y_train = np.array([[[0 if i == "M" else 1]] for i in Output])
# print (len(x_train))

# network
net = Network()
net.add(FCLayer(30, 60))
net.add(ActivationLayer(tanh, tanh_prime))
net.add(FCLayer(60, 20))
net.add(ActivationLayer(tanh, tanh_prime))
net.add(FCLayer(20, 1))
net.add(ActivationLayer(tanh, tanh_prime))

# train
net.use(mse, mse_prime)
net.fit(x_train, y_train, epochs=1000, learning_rate=0.005)

# test
out = net.predict(x_train)
# print(round(out[0][0][0]))
res = ["M" if round(i[0][0]) == 0 else "B" for i in out]
counter = 0
for i,value in enumerate(res):
    if value == Output[i]:
        counter += 1
print(Network.err_list)
plt.plot(Network.err_list, label = "rms error")
plt.legend()
plt.show()
print("accuracy:",(counter/len(res)*100))