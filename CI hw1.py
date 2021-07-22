
x = [
95,95,95,95,148,149,150,150,153,
95,95,95,149,150,150,150,153,153,
95,95,95,95,150,150,150,150,153,
95,95,95,95,150,150,150,150,153
]

def makedata(x, window=4):
  X1 = []
  X2 = []
  Y = []
  for i in range(int(len(x)/9)):
    X1.append(x[(i*9):(i*9)+window])
    X2.append(x[(i*9)+window:(i*9)+2*window])
    Y.append(x[i*9+2*window])

  return X1, X2, Y

X1,X2,Y = makedata(x)
print(X1)
print(X2)
print(Y)