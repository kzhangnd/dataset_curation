import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import argparse
from os import path, makedirs
from multiprocessing import Pool
import os
from scipy.spatial import distance

name1 = None
name2 = None
PROBE_FILE = None
PROBE = None
PROBE_O = None
GALLERY_FILE = None
GALLERY = None
TWINS = None
ID_SIZE = None
DATASET = None
METRIC = None





def match_features(output, group):
    authentic_save = path.join(output, '{}_authentic.txt'.format(group))
    impostor_save = path.join(output, '{}_impostor.txt'.format(group))
    twins_save = path.join(output, '{}_twins.txt'.format(group))
    labels_save = path.join(output, '{}_labels.txt'.format(group))

    impostor_file = open(impostor_save, 'w')
    authentic_file = open(authentic_save, 'w')
    labels_file = []



    if DATASET == 'ND':
        twins_file = open(twins_save, 'w')

        # run this in multiple processes to speed things up
    pool = Pool(os.cpu_count())

    for authentic, impostor, twins, label in pool.imap_unordered(match, PROBE):
        if impostor.shape[0] > 0:
            np.savetxt(impostor_file, impostor, delimiter=' ', fmt='%i %i %s')

        if authentic.shape[0] > 0:
            np.savetxt(authentic_file, authentic, delimiter=' ', fmt='%i %i %s')

        if twins.shape[0] > 0:
            np.savetxt(twins_file, twins, delimiter=' ', fmt='%i %i %s')

        if label is not None:
            labels_file.append(label)

    impostor_file.close()
    authentic_file.close()
    labels_file = np.array(labels_file)
    np.savetxt(labels_save, labels_file, delimiter=' ', fmt='%s')

    if DATASET == 'ND':
        twins_file.close()

    pool.close()


def chisquare(p, q):
    p = np.asarray(p).flatten()
    q = np.asarray(q).flatten()
    bin_dists = (p - q)**2 / (p + q + np.finfo('float').eps)
    return np.sum(bin_dists)


def match(x, y):
    

    image_a_path = x
    image_a = path.split(image_a_path)[1]

    features_a = np.load(image_a_path)

    if np.ndim(features_a) == 1:
        features_a = features_a[np.newaxis, :]



    image_b_path = y
    image_b = path.split(image_b_path)[1]
            
    features_b = np.load(image_b_path)

    if np.ndim(features_b) == 1:
                features_b = features_b[np.newaxis, :]

    if METRIC == 1:
        score = np.mean(cosine_similarity(features_a, features_b))
    elif METRIC == 2:
        score = distance.euclidean(features_a, features_b)
    else:
        score = chisquare(features_a, features_b)

    return score


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Match Extracted Features')
    parser.add_argument('-probe', '-p', help='Probe image list.', default = "../Shared/vggface2_test/stage_4/stage_4_begin.txt")
    parser.add_argument('-gallery', '-g', help='Gallery image list.')
    parser.add_argument('-output', '-o', help='Output folder.', default = "../Shared/AFD/s_match")
    parser.add_argument('-dataset', '-d', help='Dataset name.', default='AFD')
    parser.add_argument('-subject', '-s', help='Subject List', default="../Shared/vggface2_test/stage_4/stage_4_subject_begin.txt")
    parser.add_argument('-group', '-gr', help='Group name, e.g. AA')
    parser.add_argument('-metric', '-m', default=1,
                        help='Metric to us: (1) Cosine Similarity; (2) Euclidean Distance; (3) Chi Square')

    args = parser.parse_args()

    if args.gallery is None:
        args.gallery = args.probe

    if not path.exists(args.output):
        makedirs(args.output)

    DATASET = args.dataset.upper()
    METRIC = int(args.metric)

    if DATASET == 'ND':
        TWINS = np.loadtxt(
            '../Balanced_MTCNN/twins.txt', delimiter=' ', dtype=np.str)
        ID_SIZE = 9
    elif DATASET == 'MORPH':
        ID_SIZE = 6
    elif DATASET == 'IJBB':
        ID_SIZE = -1
    elif DATASET == 'AFD':
        ID_SIZE = -1
    else:
        raise Exception('NO FILE PATTERN FOR THE DATASET INFORMED.')

    PROBE_FILE = args.probe
    PROBE_O = np.sort(np.loadtxt(PROBE_FILE, dtype=np.str))

    GALLERY_FILE = args.gallery
    GALLERY = np.sort(np.loadtxt(args.gallery, dtype=np.str))

    subject = np.sort(np.loadtxt(args.subject, dtype=np.str, delimiter=' '))

    deletion = []


    for i in range(len(subject)):

        sign = []

        t = subject[i]

        for x in PROBE_O:
            sb = x.split('/')[10]
            if sb == t:
                sign.append(x)

        for j in range(len(sign)-1):

            if sign[j] not in deletion:

                name1 = sign[j].split('/')[11]

                for k in range(j+1, len(sign)):

                    if sign[k] not in deletion:
                        
                        name2 = sign[k].split('/')[11]

                        result = match(sign[j], sign[k])

                        if result >= 0.91:

                            deletion.append(sign[k])


            np.savetxt('../Shared/vggface2_test/stage_4/deletion.txt', deletion, delimiter=' ', fmt='%s')



        

    

    

