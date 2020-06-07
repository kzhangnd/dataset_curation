import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import argparse
from os import path, makedirs
from datetime import datetime
from tqdm import tqdm


PROBE_FILE = None
PROBE = None
PROBE_O = None
METRIC = None


def skip_diag(A):   #remove the diagonal element of a matrix
    m = A.shape[0]
    strided = np.lib.stride_tricks.as_strided
    s0, s1 = A.strides
    return strided(A.ravel()[1:], shape=(m-1,m), strides=(s0 + s1, s1)).reshape(m, -1)

class Matcher():
    def __init__(self, probe_list):
        # load features from probe file
        self.probe = np.asarray([x for x in map(np.load, probe_list)])

        # initiate a matrix NxM with zeros representing authentic matches
        self.length = len(self.probe)
        self.authentic_impostor = np.zeros(shape=(self.length, self.length))
        for i in range(len(self.probe)):
            # remove same feature files
            self.authentic_impostor[i, i] = -1

        self.matches = None

    def match_features(self):
        self.matches = cosine_similarity(self.probe, self.probe)

    def remove_mislabelled(self):
        x = np.arange(self.length)
        if METRIC == 1:
            score = np.transpose(np.vstack([x, np.mean(skip_diag(self.matches), axis=1)]))
        else:
            score = np.transpose(np.vstack([x, np.median(skip_diag(self.matches), axis=1)]))
        
        score = score[score[:, 1].argsort()]    # sort the score
        gap = np.diff(score[:, 1])  # calculate the gap
        return score[np.argmax(gap)+1:, 0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Match Extracted Features')
    parser.add_argument('-probe', '-p', help='Probe image list.', default = "../Shared/AS/stage_1/all.txt")
    parser.add_argument('-metric', '-m', default=2,
                        help='Metric to us: (1) Mean; (2) Median')

    args = parser.parse_args()
    METRIC = args.metric

    print("Loading files ...")
    PROBE = np.sort(np.loadtxt(args.probe, dtype=np.str))
    print("Finished loading ...")

    dic = {}
    for line in tqdm(PROBE):
        subject = line.split('/')[-2]
        if subject not in dic:
            dic[subject] = []
        dic[subject].append(line)

    result = []
    for probe_list in tqdm(dic.values()):
        matcher = Matcher(probe_list)
        matcher.match_features()
        index = matcher.remove_mislabelled()
        result.extend([probe_list[int(i)] for i in index])


    np.savetxt('../Shared/AS/stage_2/median.txt', result, delimiter=' ', fmt='%s')



        