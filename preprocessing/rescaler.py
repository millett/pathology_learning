from imageio import imread, imsave
import os
from scipy.misc import imresize
from tqdm import tqdm
import numpy as np
from multiprocessing import Pool

in_paths = ['../tcga/dense/train/gbm/','../tcga/dense/train/lgg/','../tcga/dense/val/gbm/','../tcga/dense/val/lgg/']
out_paths = ['../tcga/dense256/train/gbm/','../tcga/dense256/train/lgg/','../tcga/dense256/val/gbm/','../tcga/dense256/val/lgg/']

pairs = [(in_paths[i], out_paths[i]) for i in range(len(in_paths))]


in_path = None
out_path = None

pool = Pool(8)

def resizeFile(filename):
    global in_path
    global out_path
    
    img = np.asarray(imread(in_path + filename))
    resized = imresize(img, (256,256))
    imsave(out_path + filename, resized)

for _in, _out in pairs:
    in_path = _in
    out_path = _out
    filenames = os.listdir(in_path)
    for _ in tqdm(pool.imap_unordered(resizeFile, filenames), total=len(filenames)):
        pass