import math
import sys
from collections import defaultdict

data = [
  ["high","vhigh","3","4","medium","high","Q"],
  ["high","medium","4","6","small","high","S"],
  ["medium","vhigh","3","1","big","medium","S"],
  ["medium","high","3","4","big","high","R"],
  ["low","high","1","4","small","low","S"],
  ["low","vhigh","3","1","big","medium","Q"],
  ["low","vhigh","1","6","small","high","P"],
  ["low","high","3","1","medium","low","S"],
  ["low","low","4","6","small","medium","Q"],
  ["vlow","low","3","1","big","low","P"],
  ["vlow","low","6","4","small","high","S"], 
  ["vlow","low","4","4","big","medium","R"]
]

def entropy(classCounts):
  res = 0
  total = sum(classCounts)
  for i in classCounts:
    if i != 0:
      res += (i/total)*math.log2(i/total)
  return -res

def infomation(data):
  res = 0
  total = 0
  for i in data:
    total += sum(i)
  for i in data:
    res += (sum(i)/total)*entropy(i)
  return res

def getLabelCounts(data):
  counts = defaultdict(int)
  for i in data:
    counts[i[-1]] += 1
  return counts

def getFeatureSplit(data,featureIndex):
  c = getLabelCounts(data)
  res = {}
  d = {}
  for i,k in enumerate(list(c)):
    d[k] = i
  for instance in data:
    if instance[featureIndex] in res:
      res[instance[featureIndex]][d[instance[-1]]] += 1
    else:
      l = [0 for _ in c.keys()]
      l[d[instance[-1]]] = 1
      res[instance[featureIndex]] = l
  return res,d
      

# returns the index of the feature to split on
# prints intermediate steps
def splitOnFeatures(data, useGainRatio=False):
  """
  input: data as a list of lists, useGainRatio as a bool
  output: the feature to split on, and the max entropy gain
  """
  counts = getLabelCounts(data)
  baseEntropy = entropy(list(counts.values()))
  print(f'Base Entropy: {baseEntropy}')
  
  maxEntropyGain = 0
  featureToSplitOn = -1
  for i in range(len(data[0])-1):
    print(f'Feature {i+1}:')
    splits, labels = getFeatureSplit(data,i)
    print(f'Labels (class: index): {labels}')
    print(f'Split made: {splits}')
    l = list(splits.values())
    splitInfo = entropy([sum(s) for s in l])
    print(f'Split Info: {splitInfo}')
    info = infomation(l)
    print(f'Info: {info}')
    gain = baseEntropy - info
    if useGainRatio:
      gain = gain / splitInfo
      
    print(f'Gain: {gain}')
    print()
    if gain > maxEntropyGain:
      maxEntropyGain = gain
      featureToSplitOn = i
  return maxEntropyGain, featureToSplitOn
    
sys.stdout = open('./standard2.txt','w')
print('Without using gain ratio criteria:')
maxEntropyGain, featureToSplitOn = splitOnFeatures(data)
print(f'Splitting on feature {featureToSplitOn + 1} with gain of {maxEntropyGain}')
print()
print('-----------------------------------------------------------------------------')
print()
print('Using gain ratio criteria:')
maxEntropyGain, featureToSplitOn = splitOnFeatures(data,useGainRatio=True)
print(f'Splitting on feature {featureToSplitOn + 1} with gain of {maxEntropyGain}')

# print(splitOnFeatures(data,useGainRatio=True))