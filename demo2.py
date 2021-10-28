a = [int(n) for n in input().split()]
b=int(input())
max=-2000000                                                                                    
index=0
for i in range (b-1,len(a)):
    if sum(a[i-b+1:i])>max:
        max = sum(a[i-b+1:i])
        index = i-b+1
print(max)
print(index)