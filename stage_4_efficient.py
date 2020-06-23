import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import argparse
from os import path, makedirs
from datetime import datetime
from tqdm import tqdm

def get_label(p):
    return p.split('/')[-2] + '/' + p.split('/')[-1][:-4]

class Matcher():
    def __init__(self, probe_list):
        # load features from probe file
        self.probe = np.asarray([x for x in map(np.load, probe_list)])
        self.label = list(map(get_label, probe_list))
        self.length = len(self.probe)

        self.matches = None

    def match_features(self):
        self.matches = cosine_similarity(self.probe, self.probe)

    def remove_duplicate(self):
        score = []
        for i in range(self.length):
            for j in range(i+1, self.length):
                if self.matches[i, j] >= 0.85:
                    score.append([self.label[i], self.label[j], self.matches[i, j]])

        return score


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Match Extracted Features')
    parser.add_argument('-probe', '-p', help='Probe image list.', default = "../Shared/AS/stage_3/feature_list.txt")
    args = parser.parse_args()

    print("Loading files ...")
    # open the file to avoid length error
    dic = {}
    with open(args.probe) as f:
        for line in f:
            curr = line.strip()
            subject = curr.split('/')[-2]
            if subject not in dic:
                dic[subject] = []
            dic[subject].append(curr)
    print("Finished loading ...")

    result = []
    for probe_list in tqdm(dic.values()):
        matcher = Matcher(probe_list)
        matcher.match_features()
        result.extend(matcher.remove_duplicate())

    np.savetxt('../Shared/AS/stage_4/pairs.txt', result, delimiter=' ', fmt='%s')
    



        