import os
import numpy as np
from tqdm import tqdm

val_proportion = 0.1

imagenames = os.listdir('../tcga/originals/gbm')
imagepaths = ['../tcga/dense_full_mag/gbm/' + image_name + "*" for image_name in imagenames]
chosen_val_paths = np.random.choice(imagepaths, size=int(len(imagenames) * val_proportion))

for path in tqdm(chosen_val_paths):
    os.system('ls ' + path + '| xargs mv -t ../tcga/dense_full_mag/val/gbm/')