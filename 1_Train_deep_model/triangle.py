from skimage import filters
import numpy as np
import glob
from utils import PSNR, GeneratorEnqueuer, DirectoryIterator_DIV2K, _load_img_array, _rgb2ycbcr,SSIM, plot_dice, plot_ssim, plot_loss, DiceLoss,add_batch  
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score
from sklearn.metrics import f1_score 
from PIL import Image
import os
import sys
from os.path import join, isdir
from os import mkdir
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from scipy import ndimage as ndi
from skimage.measure import label
from skimage.filters import threshold_niblack, threshold_triangle
VERSION = "triangle"


class Print_Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

if not isdir('logger'):
    mkdir('logger')
if not isdir('logger/{}'.format(str(VERSION))):
    mkdir('logger/{}'.format(str(VERSION)))
sys.stdout = Print_Logger(os.path.join('logger/{}'.format(str(VERSION)), 'log.txt')) 

TEST_DIR = './test/'   
# Test for validation images
files_gt = glob.glob(TEST_DIR + 'mask/*.png')
files_gt.sort()
files_lr = glob.glob(TEST_DIR + 'image/*.png')
files_lr.sort()

dice_test = []
accuracies = []
sensitivities = []
specificities = []


for ti, fn in enumerate(files_gt):
    # Load HR image
    tmp = _load_img_array(files_gt[ti], 'BIN')
    val_H = np.asarray(tmp).astype(np.float32)  # HxWxC

    # Load LR image
    tmp_L = _load_img_array(files_lr[ti], 'GRAY')
    val_L = np.asarray(tmp_L).astype(np.float32)  # HxWxC

    threshold =  threshold_triangle(val_L)
    print(threshold.shape)
    binary_image = val_L>= threshold  # Convert labels to binary

    img_gt = val_H.squeeze()
    binary_image = binary_image.squeeze()

    labels, pred = add_batch(img_gt, binary_image)
    pred = pred.astype(np.float32)
    print(pred.shape)
    print(pred.max())
    print(labels.shape)
    print(labels.max())

    dice_test.append(f1_score(labels, pred))

    conf_matrix = confusion_matrix(labels, pred)
    TN, FP, FN, TP = conf_matrix.ravel()
    acc = accuracy_score(labels, pred)
    sens = recall_score(labels, pred)
    spec = TN / (TN + FP)
    accuracies.append(acc)
    sensitivities.append(sens)
    specificities.append(spec)

    image_output = ((binary_image) * 255).astype(np.uint8)
    Image.fromarray(image_output).save('result/{}/{}.png'.format(str('triangle'), fn.split('/')[-1]))

dice_avg_test = np.mean(np.asarray(dice_test))
avg_accuracy = np.mean(accuracies)
avg_sensitivity = np.mean(sensitivities)
avg_specificity = np.mean(specificities)

print('TEST AVG dice : {}'.format(dice_avg_test))
print('TEST AVG Accuracy: {}'.format(avg_accuracy))
print('TEST AVG Sensitivity: {}'.format(avg_sensitivity))
print('TEST AVG Specificity: {}'.format(avg_specificity))
