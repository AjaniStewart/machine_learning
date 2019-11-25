import arff

data = arff.load(open('/Users/ajani/wekafiles/data/project/DatasetTEST4.arff','r'))

def classify(instance):
  missing = 0
  for i in range(164,len(instance)-1):
     if instance[i] is None or instance[i] == '':
          missing += 1
  if missing > 15:
    return '1'
  return '2'

def test():
  correct = 0
  for instance in data['data']:
    if instance[203] == classify(instance):
      correct += 1
  return correct/len(data['data'])

print( test() )