import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import argparse
from os import path, makedirs
from multiprocessing import Pool
import os
from scipy.spatial import distance
from tqdm import tqdm


PROBE_FILE = None
PROBE = None
PROBE_O = None
METRIC = None


def chisquare(p, q):
    p = np.asarray(p).flatten()
    q = np.asarray(q).flatten()
    bin_dists = (p - q)**2 / (p + q + np.finfo('float').eps)
    return np.sum(bin_dists)


def match(a, file_list):
      
    value = []

    image_a_path = a

    features_a = np.load(image_a_path)

    if np.ndim(features_a) == 1:
        features_a = features_a[np.newaxis, :]

    for file_path in file_list:
    	
        image_b_path = file_path

        if image_a_path == image_b_path:
            continue
            
        features_b = np.load(image_b_path)

        if np.ndim(features_b) == 1:
            features_b = features_b[np.newaxis, :]

        if METRIC == 1:
            score = np.mean(cosine_similarity(features_a, features_b))
        elif METRIC == 2:
            score = distance.euclidean(features_a, features_b)
        else:
            score = chisquare(features_a, features_b)

        value.append(score)


    result = np.mean(np.asarray(value)) #change here for mean or median


    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Match Extracted Features')
    parser.add_argument('-probe', '-p', help='Probe image list.', default = "../Shared/AS/stage_1/all.txt")
    #parser.add_argument('-group', '-gr', help='Group name, e.g. AA')
    parser.add_argument('-metric', '-m', default=1,
                        help='Metric to us: (1) Cosine Similarity; (2) Euclidean Distance; (3) Chi Square')

    args = parser.parse_args()

    METRIC = int(args.metric)

    PROBE_FILE = args.probe

    print("Loading files ...")
    PROBE_O = np.sort(np.loadtxt(PROBE_FILE, dtype=np.str))
    print("Finished loading ...")

    dic = {}
    remain = []

    for line in tqdm(PROBE_O):
        subject = line.split('/')[-2]
        if subject not in dic:
            dic[subject] = []
        dic[subject].append(line)

    for folder in tqdm(dic.values()):
        result = []
        for x in folder:
            result.append([x, match(x, folder)])

        
        result.sort(key=lambda x: x[1])

        gap = []

        for i in range(len(result) - 1):
            gap.append(result[i + 1][1] - result[i][1])

        max_position = 0
        maximum = 0
        for i in range(len(gap)):
            if gap[i] >= maximum:
                max_position = i
                maximum = gap[i]

        for i in range(max_position+1, len(result)):
            remain.append(result[i][0])


    np.savetxt('../Shared/AS/stage_2/median.txt', remain, delimiter=' ', fmt='%s')



        

    

    

