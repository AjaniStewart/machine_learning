'''
Fixed knn classifier
it classifies only on numeric data
the data it expects is list(tuple) of lists(tuples), with the last element being the class label
that is; if i have a 10 instances with 5 features each, knn expects a 10 x 6 (5 features plus class label) nested list(tuple)

'''
#for sqrt
from math import sqrt
from collections import defaultdict

def knn(data, new_instance, k, dist_metric, normalized=False):
  #store distances from data with corresponding label, sort by distance, then take the first k distances
  #simplest method for computing but not optimally efficient
  #since it is sorted, if 2 instances have the same distance from the new instance
  #the one that is put in higher order depends on python's sorting algorithm
  #eg k = 4, distances(before sorting) = [..., (0.5, A), ..., (0.5, B), ...]
  #distances (after sorting) = [... 3 entries, (0.5, B), (0.5, A), ...]
  #B would be picked as the fourth nearest neighbour instead of A
  #the decision to do it like this reduces the amount of I have to write and because i do not presently bias one class
  #over another
  if not data:
    raise ValueError("No Data Provided")

  _data = None
  _new_instance = None
  if normalized:
    [_data, _new_instance] = normalize(data,new_instance)
  else:
    _data = data
    _new_instance = new_instance

  distances = []

  for instance in _data:
    distances.append((dist_metric(instance[:-1], _new_instance),instance[-1]))

  distances = sorted(distances, key=lambda dist: dist[0])
  distances = distances[:k]
  freqs = defaultdict(int)

  for d in distances:
    freqs[d[1]] += 1

  classes = list(freqs.keys())
  return max(classes, key=lambda c: freqs[c])

def normalize(data, new_instance):
  normalized_data = [[] for _ in range(len(data))]
  normalized_instance = []

  for cur_index in range(len(data[0])):
    # this would be the label for the feature
    if cur_index == len(data[0]) - 1:
      for index in range(len(data)):
        normalized_data[index].append(data[index][cur_index])
      break

    max = new_instance[cur_index]
    min = new_instance[cur_index]
    for feature in data:
      if feature[cur_index] > max:
        max = feature[cur_index]
      elif feature[cur_index] < min:
        min = feature[cur_index]
    for index in range(len(data)):
      normalized_data[index].append((data[index][cur_index] - min) / (max - min))
    normalized_instance.append((new_instance[cur_index] - min) / (max - min))
    
  return normalized_data, normalized_instance


def manhattan_dist(fv1,fv2):
  if len(fv1) != len(fv2):
    return None
  sum = 0
  for i in range(len(fv1)):
    sum += abs(fv1[i]-fv2[i])
  return round(sum, 4)

def euclidean_dist(fv1,fv2):
  if len(fv1) != len(fv2):
    return None
  sum_sq = 0
  for i in range(len(fv1)):
    sum_sq += (fv1[i]-fv2[i])**2

  return round(sqrt(sum_sq), 4)

if __name__ == "__main__":
  data = [
  [5.1, 3.5, 1.3, 0.2,'A'], [4.7, 3.1, 1.6, 0.2,'A'], [6.5, 2.7, 4.6, 1.5,'A'], [4.7, 2.4, 3.2, 1,'A'], [6, 2.2, 5, 1.4,'A'],
  [5, 4.6, 1.4, 0.1,'B'], [5.7, 2.8, 4.4, 1.3,'B'], [7.7, 3.6, 6.7, 2.2,'B'], [6.9, 3.2, 5.7, 9.3,'B'],
  [5.7, 3.4, 1.3, 0.2,'C'], [5.9, 3.3, 4, 1.3,'C'], [6.3, 3.3, 4.7, 1.4,'C'],[7.7, 2.6, 6.5, 2.3,'C'], [5.6, 2.2, 4.6, 2,'C']
  ]

  ni = [5,2.8,4.6,0.7]
  print(knn(data,ni,1,euclidean_dist,normalized=True))
  print(knn(data,ni,4,euclidean_dist,normalized=True))
  print(knn(data,ni,6,euclidean_dist,normalized=True))
  print(knn(data,ni,1,manhattan_dist,normalized=True))
  print(knn(data,ni,4,manhattan_dist,normalized=True))
  print(knn(data,ni,6,manhattan_dist,normalized=True))
  print(knn(data,ni,1,euclidean_dist))
  print(knn(data,ni,4,euclidean_dist))
  print(knn(data,ni,6,euclidean_dist))
  print(knn(data,ni,1,manhattan_dist))
  print(knn(data,ni,4,manhattan_dist))
  print(knn(data,ni,6,manhattan_dist))
    
