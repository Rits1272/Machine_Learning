from math import sqrt
import numpy as np
from collections import Counter
import warnings
import pandas as pd 
import random

# [[plt.scatter(ii[0], ii[1], s=100, color=i) for ii in dataset[i]] for i in dataset]
# plt.scatter(new_features[0], new_features[1], s=100)
# plt.show()

def k_nearest_neighbors(data, predict, k=3):
	if len(data) >= k:
		warnings.warn('K is set to a value less than total voting groups')

	distances = []
	
	for group in data:
		for features in data[group]:
			#euclidean_distance = sqrt( (features[0] - predict[0]) ** 2 + (features[1] - predict[1]) ** 2 )	
			#euclidean_distance = np.sqrt(np.sum((np.array(features)) - np.array(predict)) ** 2) # In numpy
			euclidean_distance = np.linalg.norm(np.array(features) - np.array(predict))  # Simplest way
			distances.append([euclidean_distance, group])

		votes = [i[1] for i in sorted(distances)[:k]]  # eg. votes=['r','k','r',...]
		vote_result = Counter(votes).most_common(1)[0][0]
		confidence = Counter(votes).most_common(1)[0][1] / k

	return vote_result, confidence

# result = k_nearest_neighbors(dataset, new_features, k=3)
# print(result)

# [[plt.scatter(ii[0], ii[1], s=100, color=i) for ii in dataset[i]] for i in dataset]
# plt.scatter(new_features[0], new_features[1], s=100, color=result)
# plt.show()

df = pd.read_csv('./Dataset/breast-cancer-wisconsin.data')
df.replace('?', -99999, inplace=True)
df.drop(['id'], 1, inplace=True)
full_data = df.astype(float).values.tolist()
random.shuffle(full_data)

# Our train_test_split

test_size = 0.2
train_data = full_data[:-int(test_size * len(full_data))]
test_data = full_data[-int(test_size * len(full_data)):]
test_set = {2:[], 4:[]}
train_set = {2:[], 4:[]}

for i in test_data:
	test_set[i[-1]].append(i[:-1])

for i in train_data:
	train_set[i[-1]].append(i[:-1])

correct = 0
total = 0

for group in test_set:
	for data in test_set[group]:
		vote, confidence = k_nearest_neighbors(train_set, data, k=5)
		if group == vote:
			correct += 1
		total += 1

print('Accuracy: ', correct/total)

