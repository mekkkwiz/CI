a=input().split()
b=int(input())
listSum=[]
for i in range (len(a)):
  a[i]=int(a[i])
for i in range (b-1,len(a)):
    sum=0
    for j in range (b):
        sum+=a[i-j]
    pair=[sum,i-b+1]
    listSum.append(pair)
max=-2000000
index=0
for obj in listSum:
    if obj[0] > max:
        max = obj[0]
        index = obj[1]
print(max)
print(index)