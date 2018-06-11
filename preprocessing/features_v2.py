import skimage.io
import pandas as pd
import openslide
import sys
import os
import logging
import heapq
import numpy as np
import histomicstk as htk
import scipy as sp
import concurrent.futures
from tqdm import tqdm
from multiprocessing import Pool, Value
import time
import pickle
from tqdm import tqdm, trange
import histomicstk.segmentation.positive_pixel_count as ppc

LOG_FILENAME = 'failed_files.log'
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename=LOG_FILENAME,
                    filemode='w')

poolsize = 24
n_chunks = 500


argnum = len(sys.argv)
if argnum != 2:
    sys.exit('wrong number of arguments:' + str(len(sys.argv))+'. Please enter a tumor type (gbm or lgg)')
tumor_type = sys.argv[1]
if tumor_type != "gbm" and tumor_type != "lgg":
    sys.exit("Wrong type. Please enter a tumor type (gbm or lgg)")

in_paths = ["../tcga/dense/train/" + tumor_type + "/", "../tcga/dense/val/" + tumor_type + "/"]
#out_path = "../tcga/" + "dense_features/train/" + tumor_type + "/"



def getPatName(filename):
    #pat_name = 'TCGA-02-0329'
    return filename.split(tumor_type+"/")[1][:len('TCGA-CS-4938')]
def getImageID(filename):
    return getPatName(filename) + filename[-5] # -1 g -2 n -3 p -4 . -5 number!

def getOriginalName(imname):
    return imname.split("_")[0]

def chunkToImages(imnames):
    chunks = []
    for i in (range(len(imnames))):
        chunk = []
        ref_imname = imnames[i]
        original_name = getOriginalName(imnames[i])
        while i < len(imnames) and original_name in imnames[i]:
            chunk.append(imnames[i])
            i += 1
        chunks.append(chunk)
    return chunks



#adapted from the ipynb example from HistomicsTK
def extractImageFeatures(im_reference, filename):
    im_input = skimage.io.imread(filename)[:, :, :3]
    original_name = getOriginalName(filename)
    ref_image_file = (original_name + "_0.png") 
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
    min_radius = 10
    max_radius = 20

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
    min_nucleus_area = 10
    im_nuclei_seg_mask = htk.segmentation.label.area_open(im_nuclei_seg_mask, min_nucleus_area).astype(np.int)
    im_feats = htk.features.compute_morphometry_features(im_nuclei_seg_mask).mean()
    pat_name, imageIdx  = getPatName(filename), getImageID(filename)
    im_feats['Case'], im_feats['ID'] = pat_name, getImageID(filename)
    params = ppc.Parameters(
        hue_value=0.05,
        hue_width=0.1,
        saturation_minimum=0.05,
        intensity_upper_limit=1.0,
        intensity_weak_threshold=0.85,
        intensity_strong_threshold=0.15,
        intensity_lower_limit=0.0,
    )
    
    fields = ["NumberWeakPositive", "NumberPositive", "NumberStrongPositive", "IntensitySumWeakPositive", "IntensitySumPositive", "IntensitySumStrongPositive", "IntensityAverage", "RatioStrongToTotal", "IntensityAverageWeakAndPositive"]
    try:
        color_chars = ppc.count_image(im_nmzd, params)
        color_chars = color_chars[0]
        for i, field in enumerate(fields):
            im_feats[field] = color_chars[i]
    except Exception as e:
        #print('color failure with '+filename + " exception: "+str(e))
        for i, field in enumerate(fields):
            if field not in im_feats:
                im_feats[field] = float('nan')
    
    
    return pd.DataFrame(pd.Series(im_feats))

def getSemanticFeatures(filename): 
    global xlsx
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

print('chunkin')
start = time.time()
images1 = [in_paths[0] + imname for imname in os.listdir(in_paths[0])]
images2 = [in_paths[1] + imname for imname in os.listdir(in_paths[1])]
all_images = sorted(images1 + images2)
chunks = chunkToImages(all_images)

mid = time.time()

print(mid - start)
print('pooling')
#n_chunks = len(chunks)
N = len(all_images)
#pbar = tqdm(total=n_chunks)
def extractChunkFeatures(chunk):
    start = time.time()
    ref_image_file = chunk[0]
    im_reference = skimage.io.imread(ref_image_file)[:, :, :3] # normalize image to upper left chosen chunk from larger image
    all_features = []
    sem_features = getSemanticFeatures(ref_image_file)
    for i in range(len(chunk)):
        try:
            im_features = extractImageFeatures(im_reference, chunk[i])
            all_features_i = pd.concat((im_features, sem_features))
            all_features.append(all_features_i)
        except Exception as e:
            #print(e)
            #print('chunk is '+chunk[i])
            pass
    with counter.get_lock():
        chunk_len = len(chunk)
        counter.value += chunk_len
        if counter.value % chunk_len == 0:
            print(str(counter.value / float(N) * 100) + "% done")
            if counter.value % (chunk_len * 2) == 0:
                print(str(time.time() - t0) + "s elapsed")
        
                
    return all_features


def init(args):
    ''' store the counter for later use '''
    global counter
    global t0
    counter, t0 = args
counter = (Value('i', 0), time.time())

pool = Pool(poolsize, initializer=init, initargs = (counter, ))
all_data = (pool.map(extractChunkFeatures, chunks))
print(time.time() - mid)
midmid = time.time()
print(midmid - mid)
print('dumping')
with open('../tcga/dense_features/'+tumor_type + '_list_v2.pkl', 'wb') as f:
    pickle.dump(all_data, f)
print('total time',time.time() - start)