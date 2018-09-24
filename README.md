# pathology_learning
Using traditional machine learning methods (see BMI260_final_report.pdf) and deep learning methods (see CS231N_final_report.pdf) to predict clinical characteristics from [TCGA](https://cancergenome.nih.gov/) [pathology slides.](https://console.cloud.google.com/storage/browser/isb-cgc-open/NCI-GDC/legacy/TCGA)

This project pulls pathology slides (currently just of GBM and LGG tumors) from TCGA, processes them, and makes phenotypic predictions of tumor type, survival time, and tumor grade, based on images and semantic data (currently age and sex).

# Preprocessing images
Sub-images are selected from the slides in 2 different ways: preprocessing/padder.py reshapes whole-slide images into squares, padded with whitespace (TCGA Pathology slides are often heterogeneous in shape, and are rarely- if ever- square), in tcga/256 and tcga/1024, and by selecting chunks of highest cell density as described in Yu et al 2016 (https://www.nature.com/articles/ncomms12474), in tcga/dense.

## Traditional methods
The project uses traditional machine learning methods (currently linear and logistic regression) on nucleus features from the tcga/dense images, along with age and sex data. It uses HistomicsTK (https://github.com/DigitalSlideArchive/HistomicsTK) to segment nuclei and calculate shape statistics.

# Deep Learning Methods
PyTorch.ipynb uses a VGG net trained initially from ImageNet as described in (https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html). It makes predictions of tumor type from padded images.
