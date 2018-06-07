# adapted from https://jdhao.github.io/2017/11/06/resize-image-to-square-with-padding/
from PIL import Image
import openslide
import sys
import os
import logging
import heapq
import numpy as np
from tqdm import tqdm, trange
from multiprocessing import Pool
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

print("tumor type",tumor_type)
IN_PATH = "../tcga/originals/" + tumor_type + "/"
OUT_PATH = "../tcga/" + "dense_full_mag" + "/" + tumor_type + "/"
DESIRED_SIZE = 2000
overlap = 100
magnification = 0

print(55000.0 / (DESIRED_SIZE - overlap))

def calc_density(image_chunk):
	pixels = np.array(image_chunk)
	return float(np.sum(pixels < 200)) / pixels.size

#Use a min heap to keep top 10 density chunks for each image.
def selectTop10(full_path, desired_size=DESIRED_SIZE):   
	top10 = []
	original = openslide.OpenSlide(full_path)
	full_x, full_y = original.level_dimensions[magnification]
	i = 0
	for x_left in trange(0, full_x - desired_size, desired_size - overlap):
		for y_top in range(0, full_y - desired_size, desired_size - overlap):
			chunk = original.read_region(location=(x_left, y_top), level=magnification, size=(desired_size, desired_size))
			density_score = calc_density(chunk)
			entry = (density_score, i, chunk)
			if len(top10) < 10:
				heapq.heappush(top10, entry)
			elif density_score > top10[0][0]: #min density at top
				heapq.heappushpop(top10, entry)
			i+=1
	return top10

def saveTop10(filename, desired_size=DESIRED_SIZE):
	global IN_PATH
	savepath = OUT_PATH + filename + "_"
	if os.path.exists(savepath + "0.png"):
		logging.error('file_already_exists; skipping')
		return
	img_path = IN_PATH + filename
	try:
		top10 = selectTop10(img_path, desired_size)
		for j, entry in enumerate(top10):
			im = entry[2]
			im.save(savepath + str(j) + '.png', "PNG")
	except:
		print('failure')
		logging.error('tile error with '+filename)
    
filenames = os.listdir(IN_PATH)

#saveTop10(filenames[0])#'TCGA-S9-A7QZ-01A-01-TSA.BD1DB4BE-E9A6-4AB9-B873-C8B72DFBD6C9.svs')

#print(filenames)

pool = Pool(8)

for _ in tqdm(pool.imap_unordered(saveTop10, filenames), total=len(filenames)):
	pass