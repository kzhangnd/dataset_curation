import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import argparse
from os import path, makedirs
import random
from tqdm import tqdm

PROBE = None

def match(probe, gallery):
    return np.mean(cosine_similarity(probe, gallery))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Match Extracted Features')
    parser.add_argument('-probe', '-p', help='Probe image list.', default = '../Shared/AS/stage_2/union_image_final.txt')
    parser.add_argument('-output', '-o', help='Output folder.', default = '../Shared/AS/stage_3')

    args = parser.parse_args()

    if not path.exists(args.output):
        makedirs(args.output)

    print('Loading files ...')
    PROBE = np.sort(np.loadtxt(args.probe, dtype=np.str))
    print('Finished loading ...')

    dic = {}
    for line in tqdm(PROBE):
        subject = line.split('/')[-2]
        if subject not in dic:
            dic[subject] = []
        dic[subject].append(line)

    print('Start loading npy files ...')
    _map = {}
    for key, value in tqdm(dic.items()):
        _map[key] = np.asarray([x for x in map(np.load, random.sample(value, 5))])
    
    print("Iterate over the combinations ...")
    result = []
    subject_list = list(_map.keys())
    for i in tqdm(range(len(subject_list))):
        curr_subject = subject_list[i]
        curr_probe = _map[subject_list[i]]
        curr = []
        for j in range(i+1, len(subject_list)):
            curr.append([curr_subject, subject_list[j], match(curr_probe, _map[subject_list[j]])])
    
        result.extend(sorted(curr, key=lambda p: p[2], reverse=True)[:5])

    np.savetxt('../Shared/AS/stage_3/raw.txt', sorted(result, key=lambda p: p[2], reverse=True)[:5000], delimiter=' ', fmt='%s')

    