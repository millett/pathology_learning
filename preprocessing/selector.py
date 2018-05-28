# adapted from https://jdhao.github.io/2017/11/06/resize-image-to-square-with-padding/
from PIL import Image
import openslide
import sys
import os
import logging
import heapq
import numpy as np
from tqdm import trange
import concurrent.futures
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
in_path = "../tcga/originals/" + tumor_type + "/"
out_path = "../tcga/" + "dense" + "/" + tumor_type + "/"
desired_size = 1000
overlap = 200
magnification = 1

def calc_density(image_chunk):
	pixels = np.array(image_chunk)
	return float(np.sum(pixels < 200)) / pixels.size

#Use a min heap to keep top 10 density chunks for each image.
def selectTop10(full_path, desired_size):
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

filenames = os.listdir(in_path)
'''
if multithreading:
    executor = concurrent.futures.ProcessPoolExecutor(10)
    futures = [executor.submit(selectTop10, (in_path + filename, desired_size)) for filename in filenames]
    concurrent.futures.wait(futures)
    sys.exit(0)
'''

for i in trange(len(filenames)):
	filename = filenames[i]
	out_name = out_path + filename + "_" + str(2) + '.png'
	already_created = set(os.listdir(out_path))
	if out_name in already_created:
		continue
	try: 
		top10 = selectTop10(in_path + filename, desired_size)
		for j, entry in enumerate(top10):
			im = entry[2]
			im.save(out_path + filename + "_" + str(j) + '.png', "PNG")
	except:
		print("excception")
		logging.error('Tile error with '+filename)