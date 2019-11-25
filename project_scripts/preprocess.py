import arff
import pprint
from collections import defaultdict

PATH_TO_DATASET = '/Users/ajani/wekafiles/data/project/'
DATASET_NAME = 'DatasetAttrDropped3.arff'
PREPROCESSED_DATASET = 'DatasetTEST4.arff'

data = arff.load(open(PATH_TO_DATASET + DATASET_NAME,'r'))

# print(data['attributes'][203])
#RECODING: changing 999's and 997's to missing values, bereavement stuff to 777, 
# print(data['data'][643][203])

#SNL_1 needs a 7
data['attributes'][51][1].append('7')

#Bereavement stuff
for i in range(18,26):
  data['attributes'][i][1].append('777')

#???
for i in range(48,109):
  data['attributes'][i][1].append('777')
# for index, thing in enumerate(data['attributes']):
#   if thing[0].find("SNL_1") > -1:
#     print(index, thing)
for instance in data['data']:
  for i in range(len(instance)):
    if i == 51 and instance[i] == '8':
      instance[i] = '7'
    elif i == 51 and instance[i] == '9':
      instance[i] = '8'
    if str(instance[i]) == "997" or str(instance[i]) == "999" or str(instance[i]) == "997.0" or str(instance[i]) == "999.0":
      instance[i] = ''
    elif ((i > 17 and 26 > i) or (i > 47 and 109 > i)) and (instance[i] is None or str(instance[i]) == ''):
      instance[i] = '777'


missing_ones = 0
ones = 0
# d_ones = defaultdict(int)

for instance in data['data']:
  if instance[203] == '1':
      ones += 1
      for i in range(164,len(data['data'][i])-1):
        if instance[i] is None or instance[i] == '':
          missing_ones += 1
          # d_ones[data['attributes'][i][0]] += 1

missing_ones /= ones
print(f'missing ones {missing_ones} ones {ones}')
# pprint.pprint(d_ones)
# print(len(d_ones))

missing_twos = 0
twos = 0
# d_twos = defaultdict(int)

for instance in data['data']:
  if instance[203] == '2':
      twos += 1
      for i in range(164,len(data['data'][i])-1):
        if instance[i] is None or instance[i] == '':
          missing_twos += 1
          # d_twos[data['attributes'][i][0]] += 1

missing_twos /= twos
print(f'missing twos {missing_twos} twos {twos}')
# pprint.pprint(d_twos)
# print(len(d_twos))
  

# for d in data['data'][0]:
  # print(type(d))

# print((data['data'][0][5]) is None)

#DROPPING INSTANCES

removedSet = set()

averagePercentMissing = 0

for index, instance in enumerate(data['data']):
  numMissing = 0
  for i, featureVal in enumerate(instance):
    #ignore things after bditot
    if 164 > i and (featureVal is None or featureVal == ''):
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

data['data'] = newData

print(len(newData))

arff.dump(data,open(PATH_TO_DATASET + PREPROCESSED_DATASET,'w'))