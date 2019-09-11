#for sqrt
import math

data = {
  "A": [[5.1, 3.5, 1.3, 0.2], [4.7, 3.1, 1.6, 0.2], [6.5, 2.7, 4.6, 1.5], [4.7, 2.4, 3.2, 1], [6, 2.2, 5, 1.4]],
  "B": [[5, 4.6, 1.4, 0.1], [5.7, 2.8, 4.4, 1.3], [7.7, 3.6, 6.7, 2.2], [6.9, 3.2, 5.7, 9.3]],
  "C": [[5.7, 3.4, 1.3, 0.2], [5.9, 3.3, 4, 1.3], [6.3, 3.3, 4.7, 1.4],[7.7, 2.6, 6.5, 2.3], [5.6, 2.2, 4.6, 2]]
}



def knn(new_instance, k, dist_metric, normalize=False):
  #store distances from data with corresponding label, sort by distance, then take the first k distances
  #simplest method for computing but not optimally efficient
  #since it is sorted, if 2 instances have the same distance from the new instance
  #the one that is put in higher order depends on python's sorting algorithm
  #eg k = 4, distances(before sorting) = [..., (0.5, A), ..., (0.5, B), ...]
  #distances (after sorting) = [... 3 entries, (0.5, B), (0.5, A), ...]
  #B would be picked as the fourth nearest neighbour instead of A
  #the decision to do it like this reduces the amount of I have to write and because i do not presently bias one class
  #over another
  _data = None
  _new_instance = None
  if normalize:
    _data = normalize_data(data,new_instance)
    _new_instance = normalize_instance(new_instance)
  else:
    _data = data
    _new_instance = new_instance

  distances = []

  for label, fvs in _data.items():
    for feat_vec in fvs:
      distances.append((dist_metric(feat_vec, _new_instance),label))
  distances = sorted(distances, key=lambda dist: dist[0])

  distances = distances[:k]
  print("neigbors", distances, end="", file=outfile)

  freqs = {
    "A": 0,
    "B": 0,
    "C": 0
  }

  for d in distances:
    freqs[d[1]] += 1

  # picks the simple majority, if there is a class tie, pick C
  # i made this decision because, without background knowledge of the data
  # i do not know which one i pick
  # the decision to pick C is arbitrary
  if freqs["A"] > freqs["B"] and freqs["A"] > freqs["C"]:
    return "A"
  elif freqs["B"] > freqs["A"] and freqs["B"] > freqs["C"]:
    return "B"
  else:
    return "C"

def list_copy(l, num):
  return [list(l) for _ in range(num)]

def normalize_data(data, new_instance):
  normalized_data = {}
  for label, feature_vectors in data.items():
    normalized_features = list_copy([],len(feature_vectors))
    for i in range(len(feature_vectors[0])):
      #the initial value only works for the given data
      max = new_instance[i]
      min = new_instance[i]
      for feature in feature_vectors:
        if feature[i] > max:
          max = feature[i]
        elif feature[i] < min:
          min = feature[i]
      for j in range(len(feature_vectors)):
        # print("i",i,"j",j)
        normalized_features[j].append((feature_vectors[j][i] - min) / (max - min))
    normalized_data[label] = normalized_features
  return normalized_data

def normalize_instance(new_instance):
  normalized_instance = []
  feature_vectors = data["A"] + data["B"] + data["C"]
  for feature_index in range(len(feature_vectors[0])):
    max = new_instance[feature_index]
    min = new_instance[feature_index]
    for feature in feature_vectors:
      if feature[feature_index] > max:
        max = feature[feature_index]
      elif feature[feature_index] < min:
        min = feature[feature_index]
    normalized_instance.append((new_instance[feature_index] - min) / (max - min))
  return normalized_instance
    

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

  return round(math.sqrt(sum_sq), 4)


with open("out.txt", "w") as outfile:
  print("---------------Euclidean Distance----------------", file=outfile)
  print("k\t\t1\t4\t6", file=outfile)
  print("normalized: ",knn([5,2.8,4.6,0.7], 1, euclidean_dist, normalize=True), knn([5,2.8,4.6,0.7], 4, euclidean_dist, normalize=True), knn([5,2.8,4.6,0.7], 6, euclidean_dist, normalize=True), sep="\t", file=outfile)
  print("unnormalized: ",knn([5,2.8,4.6,0.7], 1, euclidean_dist), knn([5,2.8,4.6,0.7], 4, euclidean_dist), knn([5,2.8,4.6,0.7], 6, euclidean_dist), sep="\t", file=outfile)
  print("---------------Manhattan Distance----------------", file=outfile)
  print("k\t\t1\t4\t6", file=outfile)
  print("normalized: ",knn([5,2.8,4.6,0.7], 1, manhattan_dist, normalize=True), knn([5,2.8,4.6,0.7], 4, manhattan_dist, normalize=True), knn([5,2.8,4.6,0.7], 6, manhattan_dist, normalize=True), sep="\t", file=outfile)
  print("unnormalized: ",knn([5,2.8,4.6,0.7], 1, manhattan_dist), knn([5,2.8,4.6,0.7], 4, manhattan_dist), knn([5,2.8,4.6,0.7], 6, manhattan_dist), sep="\t", file=outfile)