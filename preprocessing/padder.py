# adapted from https://jdhao.github.io/2017/11/06/resize-image-to-square-with-padding/
from PIL import Image
import openslide
import sys
import os
import logging
from tqdm import trange
LOG_FILENAME = 'failed_files.log'
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename=LOG_FILENAME,
                    filemode='w')

argnum = len(sys.argv)
if argnum > 1:
	if argnum != 3:
		print(sys.argv)
		sys.exit('wrong number of arguments. Please enter a tumor type followed by desired size.')
	tumor_type = sys.argv[1]
	desired_size = int(sys.argv[2])
else:
	tumor_type = "gbm"
	desired_size = 256

in_path = "../tcga/originals/" + tumor_type + "/"
out_path = "../tcga/" + str(desired_size) + "/" + tumor_type + "/"


def makePaddedImg(full_path, desired_size):
	original = openslide.OpenSlide(full_path)
	old_size = original.dimensions
	ratio = float(desired_size / max(old_size))
	new_size = tuple([int(dim * ratio) for dim in old_size])
	resized = original.get_thumbnail(new_size)
	padded = Image.new("RGB", (desired_size, desired_size), (255, 255, 255))

	padded.paste(resized, ((desired_size-new_size[0])//2, (desired_size-new_size[1])//2))
	return padded

filenames = os.listdir(in_path)
for i in trange(len(filenames)):
	filename = filenames[i]
	try:
		im = makePaddedImg(in_path + filename, desired_size)
		im.save(out_path + filename + '.png', "PNG")
	except:
		logging.error('Tile error with '+filename)