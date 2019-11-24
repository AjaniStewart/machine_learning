import arff

PATH_TO_DATASET = '/Users/ajani/wekafiles/data/project/'
DATASET_NAME = 'DatasetAttrDropped2.arff'
PREPROCESSED_DATASET = 'DatasetTEST2.arff'

data = arff.load(open(PATH_TO_DATASET + DATASET_NAME,'r'))


#RECODING: changing 999's and 997's to missing values


# print(type(data))
for instance in data['data']:
  for i in range(len(instance)):
    if str(instance[i]) == "997" or str(instance[i]) == "999" or str(instance[i]) == "997.0" or str(instance[i]) == "999.0":
      instance[i] = ''

# for d in data['data'][0]:
  # print(type(d))

# print((data['data'][0][5]) is None)

#DROPPING INSTANCES

removedSet = set()

averagePercentMissing = 0

for index, instance in enumerate(data['data']):
  numMissing = 0
  for featureVal in instance:
    if featureVal is None or featureVal == '':
      numMissing += 1
  # print(numMissing)
  percentMissing = numMissing / len(instance)
  averagePercentMissing += percentMissing
  if percentMissing > 0.2:
    removedSet.add(index)

# print(removedSet)
averagePercentMissing /= len(data['data'])

print(averagePercentMissing)
newData = [data['data'][i] for i in range(len(data['data'])) if i not in removedSet]

print(len(newData))
    
arff.dump(data,open(PATH_TO_DATASET + PREPROCESSED_DATASET,'w'))