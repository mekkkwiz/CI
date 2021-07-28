import numpy as np
from random import random
from fillspace import X1, X2, Y

# items = np.array([X1[i]+X2[i] for i in range(len(Y))])
# print(items)
# targets = np.array([Y[i] for i in range(len(Y))])
# print(targets)
x = X1
normalized = [((x[i]-min(x))/(max(x)-min(x))) for i in range(len(x))]
print(normalized)