import numpy as np
import matplotlib.pyplot as plt
from random import random
from fillspace import X1, X2, Y

# items = np.array([X1[i]+X2[i] for i in range(len(Y))])
# print(items)
# targets = np.array([Y[i] for i in range(len(Y))])
# print(targets)
# x = X1
# normalized = [((x[i]-min(x))/(max(x)-min(x))) for i in range(len(x))]
# print(normalized)

# plt.plot([1, 2, 3, 4])
# plt.ylabel('some numbers')
# plt.show()

inputs = np.array([[random() / 2 for _ in range(2)] for _ in range(1000)])
target = np.array([[i[0] + i[1]] for i in inputs])
print(inputs)
print(target)