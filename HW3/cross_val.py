from random import randrange
from fillspace import *

# Split a dataset into k folds
def cross_validation_split(dataset, folds=3):
	dataset_split = list()
	dataset_copy = list(dataset)
	fold_size = int(len(dataset) / folds)
	for i in range(folds):
		fold = list()
		while len(fold) < fold_size:
			index = randrange(len(dataset_copy))
			fold.append(dataset_copy.pop(index))
		dataset_split.append(fold)
	return dataset_split

# test cross validation split
dataset = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]
# folds = cross_validation_split(m_x, 10)
# print(folds)
# print(len(folds[0][0]))
# print(folds[0][0][0])