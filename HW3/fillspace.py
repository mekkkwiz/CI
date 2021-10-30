ID = []
Input = []
Output = []

def denomallize_list(x,ref=Output):
  return [x[i] * (max(ref) - min(ref)) + min(ref) for i in range(len(x))]

def denomallize(x,ref=Output):
  return x * (max(ref) - min(ref)) + min(ref)


def normalize_list(x,ref=Output):
  return [((x[i]-min(ref))/(max(ref)-min(ref))) for i in range(len(x))]

def normalize(x,ref=Output):
  return ((x-min(ref))/(max(ref)-min(ref)))

data = []
f = open("data.txt", "r")
data = list(f.read().split('\n'))
f.close()
# # print(data)
# print(data[0])

# create array for store data
Dataset = []
# DATA
for i in range(len(data)):
    str = list(data[i].split(','))
    Dataset.append(str)

# print(Dataset)


for i in Dataset:
  ID.append(i[0])
  Output.append(i[1])
  temp = i[2:]
  Input.append([float(i) for i in temp])

# print(*Input, sep = '\n')

tempForNomalize = [[Input[j][i] for j in range(len(Input))] for i in range(30)]
nomalized_input_colum_form = [normalize_list(tempForNomalize[i],ref = tempForNomalize[i]) for i in range(30)]
nomalized_input = [[nomalized_input_colum_form[j][i] for j in range(30)] for i in range(len(Input))]

print(nomalized_input[0])





