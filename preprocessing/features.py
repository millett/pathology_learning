import skimage.io
import pandas as pd
import openslide
import sys
import os
import logging
import heapq
import numpy as np
from tqdm import trange
import histomicstk as htk
import scipy as sp
import concurrent.futures
from tqdm import tqdm
from multiprocessing import Pool, Value
import time
import pickle

LOG_FILENAME = 'failed_files.log'
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename=LOG_FILENAME,
                    filemode='w')

argnum = len(sys.argv)
if argnum != 2:
    sys.exit('wrong number of arguments:' + str(len(sys.argv))+'. Please enter a tumor type (gbm or lgg)')
tumor_type = sys.argv[1]
if tumor_type != "gbm" and tumor_type != "lgg":
    sys.exit("Wrong type. Please enter a tumor type (gbm or lgg)")

in_path = "../tcga/dense/" + tumor_type + "/"
out_path = "../tcga/" + "dense_features/" + tumor_type + "/"

images = [in_path + imname for imname in os.listdir(in_path)]

def getPatName(filename):
    #pat_name = 'TCGA-02-0329'
    return filename.split(tumor_type+"/")[1][:len('TCGA-CS-4938')]
def getImageID(filename):
    return getPatName(filename) + filename[-5] # -1 g -2 n -3 p -4 . -5 number!

#adapted from the ipynb example from HistomicsTK
def extractImageFeatures(filename):
    im_input = skimage.io.imread(filename)[:, :, :3]
    original_name = filename.split("_")[0]
    
    ref_image_file = (original_name + "_0.png") #normalize image to random set from larger image

    im_reference = skimage.io.imread(ref_image_file)[:, :, :3]

    # get mean and stddev of reference image in lab space
    mean_ref, std_ref = htk.preprocessing.color_conversion.lab_mean_std(im_reference)

    # perform reinhard color normalization
    im_nmzd = htk.preprocessing.color_normalization.reinhard(im_input, mean_ref, std_ref)

    # create stain to color map
    stainColorMap = {
        'hematoxylin': [0.65, 0.70, 0.29],
        'eosin':       [0.07, 0.99, 0.11],
        'dab':         [0.27, 0.57, 0.78],
        'null':        [0.0, 0.0, 0.0]
    }

    # specify stains of input image
    stain_1 = 'hematoxylin'   # nuclei stain
    stain_2 = 'eosin'         # cytoplasm stain
    stain_3 = 'null'          # set to null of input contains only two stains

    # create stain matrix
    W = np.array([stainColorMap[stain_1],
                  stainColorMap[stain_2],
                  stainColorMap[stain_3]]).T

    # perform standard color deconvolution
    im_stains = htk.preprocessing.color_deconvolution.color_deconvolution(im_input, W).Stains

    # get nuclei/hematoxylin channel
    im_nuclei_stain = im_stains[:, :, 0]

    # segment foreground
    foreground_threshold = 60

    im_fgnd_mask = sp.ndimage.morphology.binary_fill_holes(
        im_nuclei_stain < foreground_threshold)

    # run adaptive multi-scale LoG filter
    min_radius = 5
    max_radius = 15

    im_log_max, im_sigma_max = htk.filters.shape.cdog(
        im_nuclei_stain, im_fgnd_mask,
        sigma_min=min_radius * np.sqrt(2),
        sigma_max=max_radius * np.sqrt(2)
    )

    # detect and segment nuclei using local maximum clustering
    local_max_search_radius = 10

    im_nuclei_seg_mask, seeds, maxima = htk.segmentation.nuclear.max_clustering(
        im_log_max, im_fgnd_mask, local_max_search_radius)

    # filter out small objects
    min_nucleus_area = 5
    im_nuclei_seg_mask = htk.segmentation.label.area_open(im_nuclei_seg_mask, min_nucleus_area).astype(np.int)
    im_feats = htk.features.compute_morphometry_features(im_nuclei_seg_mask).mean()
    pat_name, imageIdx  = getPatName(filename), getImageID(filename)
    im_feats['Case'], im_feats['ID'] = pat_name, getImageID(filename)
    return pd.DataFrame(pd.Series(im_feats))

def getSemanticFeatures(filename, xlsx): 
    pat_name = getPatName(filename)
    pat_stats = xlsx[xlsx['Case'] == pat_name]
    return pat_stats[['Case','Grade', 'Age (years at diagnosis)', \
           'Gender', 'Survival (months)', 'Vital status (1=dead)',\
           'MGMT promoter status']].squeeze()


xlsx = pd.read_excel('../tcga/TableS1.PatientData.20151020.v3.xlsx', skiprows=1)
xlsx['Grade'] = xlsx['Grade'].str.slice(1)
xlsx['Grade'] = pd.to_numeric(xlsx['Grade'])
xlsx['Gender'] = (xlsx['Gender'] == 'male').astype(np.int64)
xlsx['MGMT promoter status'] = (xlsx['MGMT promoter status'] == 'Methylated').astype(np.int64)

print("starting")
data = []

N = 10000 # 1000
chunk_len = 500 # 50



n_chunks = N / chunk_len

all_data = [None] * n_chunks

print('yay images', N)

def get_data_chunk(subarray):
    global counter
    global t0
    data = []
    for i, image in enumerate((subarray)):      
        try: 
            im_features = extractImageFeatures(image)
            sem_features = getSemanticFeatures(image, xlsx)
            row = pd.concat((im_features, sem_features))
            data.append(row)
        except:
            pass
        with counter.get_lock():
            counter.value += 1
            if counter.value % chunk_len == 0:
                print(counter.value / float(N) * 100, "% done")
                if counter.value % (chunk_len * 2) == 0:
                    print(str(time.time() - t0) + "s elapsed")
    print(len(data))
    return data

chunks = [images[i:i+chunk_len] for i in range(0, N, chunk_len)]
poolsize = 8

def init(args):
    ''' store the counter for later use '''
    global counter
    global t0
    counter, t0 = args
counter = (Value('i', 0), time.time())

#t0 = time.time()
#print('chunks',np.array(chunks).shape, 'and N=',N)
pool = Pool(poolsize, initializer=init, initargs = (counter, ))
all_data = pool.map(get_data_chunk, chunks)
#all_data = pool.imap(get_data_chunk, chunks)
pool.close() # No more work
pool.join() # Wait for completion


with open('../tcga/dense_features/'+tumor_type + '_list_v1.pkl', 'wb') as f:
    pickle.dump(all_data, f)

'''
print("Starting....")
executor = concurrent.futures.ProcessPoolExecutor(10)
im_features = [executor.submit(extractImageFeatures, image) for image in images[:30]]
concurrent.futures.wait(futures)
print("And done!")
sys.exit(0)
'''