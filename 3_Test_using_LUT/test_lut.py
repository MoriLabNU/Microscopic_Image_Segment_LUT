"""Batch evaluation of segmentation using a precomputed LUT."""

from PIL import Image
import numpy as np
from os import mkdir
from os.path import isdir
from tqdm import tqdm
import glob
from sklearn.metrics import f1_score
import sys
import os
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score

from utils import _load_img_array, add_batch
import config
from lut_core import load_lut, apply_lut_segmentation

UPSCALE = config.UPSCALE
SAMPLING_INTERVAL = config.SAMPLING_INTERVAL
LUT_PATH = config.LUT_PATH
TEST_DIR = config.TEST_DIR
VERSION = config.VERSION


class Print_Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


if not isdir("logger"):
    mkdir("logger")
if not isdir("logger/{}".format(str(VERSION))):
    mkdir("logger/{}".format(str(VERSION)))
sys.stdout = Print_Logger(
    os.path.join("logger/{}/".format(str(VERSION)), "result_LUT_test.txt")
)

LUT = load_lut(LUT_PATH, upscale=UPSCALE)

files_lr = glob.glob(TEST_DIR + "/image/*.png")
files_lr.sort()
files_gt = glob.glob(TEST_DIR + "/mask/*.png")
files_gt.sort()

dice_test = []
accuracies = []
sensitivities = []
specificities = []

for ti, fn in enumerate(tqdm(files_gt)):
    img_lr = Image.open(files_lr[ti]).convert("L")
    img_lr = np.asarray(img_lr, dtype="float32")
    img_lr = img_lr[:, :, np.newaxis]

    img_gt = _load_img_array(files_gt[ti], "BIN")
    img_gt = np.asarray(img_gt, dtype="uint8")

    img_out = apply_lut_segmentation(
        img_lr, LUT, SAMPLING_INTERVAL, upscale=UPSCALE
    )

    if img_gt.shape[0] < img_out.shape[0]:
        img_out = img_out[: img_gt.shape[0]]
    if img_gt.shape[1] < img_out.shape[1]:
        img_out = img_out[:, : img_gt.shape[1]]
    if img_gt.shape[0] > img_out.shape[0]:
        img_out = np.pad(
            img_out, ((0, img_gt.shape[0] - img_out.shape[0]), (0, 0))
        )
    if img_gt.shape[1] > img_out.shape[1]:
        img_out = np.pad(
            img_out, ((0, 0), (0, img_gt.shape[1] - img_out.shape[1]))
        )

    labels, output = add_batch(img_gt, img_out)
    pred = output >= 0.5
    dice_test.append(f1_score(labels, pred))

    conf_matrix = confusion_matrix(labels, pred)
    TN, FP, FN, TP = conf_matrix.ravel()
    accuracies.append(accuracy_score(labels, pred))
    sensitivities.append(recall_score(labels, pred))
    specificities.append(TN / (TN + FP))

    image_output = ((output >= 0.5) * 255).astype(np.uint8)
    out_dir = "./output_Seg2_LUT_HV_2_test_{}bit/".format(SAMPLING_INTERVAL)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    Image.fromarray(image_output).save(
        "{}{}_LUT.png".format(out_dir, fn.split("/")[-1][:-4])
    )

dice_avg_test = np.mean(np.asarray(dice_test))
print("TEST AVG dice : {}".format(dice_avg_test))
print("TEST AVG Accuracy: {}".format(np.mean(accuracies)))
print("TEST AVG Sensitivity: {}".format(np.mean(sensitivities)))
print("TESTAVG Specificity: {}".format(np.mean(specificities)))
