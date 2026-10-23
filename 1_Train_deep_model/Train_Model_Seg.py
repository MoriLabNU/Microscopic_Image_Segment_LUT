import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from PIL import Image
import numpy as np
import time
from os import mkdir
from os.path import join, isdir
from tqdm import tqdm
import glob
from PIL import Image
from sklearn.metrics import f1_score 
from utils import PSNR, GeneratorEnqueuer, DirectoryIterator_DIV2K, _load_img_array, _rgb2ycbcr,SSIM, plot_dice, plot_ssim, plot_loss, DiceLoss,add_batch  
from tensorboardX import SummaryWriter #####################tensorboard --logdir=./
import os
import sys
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score


USE_WANDB = os.environ.get("WANDB_MODE", "").lower() != "disabled"
if USE_WANDB:
    import wandb
    _wandb_kwargs = {"project": os.environ.get("WANDB_PROJECT", "Seg_HV")}
    _wandb_entity = os.environ.get("WANDB_ENTITY")
    if _wandb_entity:
        _wandb_kwargs["entity"] = _wandb_entity
    wandb.init(**_wandb_kwargs)
else:
    wandb = None


### USER PARAMS ###
EXP_NAME = "SEG-LUT"
VERSION = "SegNet2_HV_1"

NB_BATCH = 32        # mini-batch
CROP_SIZE = 96       # input LR training patch size

START_ITER = 0      # Set 0 for from scratch, else will load saved params and trains further
#NB_ITER = 200000    # Total number of training iterations
NB_ITER = 300000
I_DISPLAY = 1000   # display info every N iteration
I_VALIDATION = 1000  # validate every N iteration
I_SAVE = 1000     # save models every N iteration
BEST_MODEL = '125000'
ONLYTEST =True  
TRAIN_DIR = './train/'  # Training images: png files should just locate in the directory (eg ./train/img0001.png ... ./train/img0800.png)
VAL_DIR = './val/'      # Validation images
TEST_DIR = './test/'      # Validation images
LR_G = 1e-4         # Learning rate for the generator
TEST = True
### Tensorboard for monitoring ###
writer = SummaryWriter(log_dir='./log/{}'.format(str(VERSION)))

class Print_Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

class CustomReflectionPad2d(nn.ReflectionPad2d):
    def forward(self, input):
        if input.shape[-1] == 1 and input.shape[-2] == 1:
            # 对于1x1的输入，使用上采样将其扩大到3x3
            return F.interpolate(input, size=(3, 3), mode='nearest')
        else:
            # 对于其他输入，使用正常的反射填充
            return super().forward(input)

# A lightweight deep network ###
# class SegNet(torch.nn.Module):
#     def __init__(self, num_classes=1):
#         super(SegNet, self).__init__()
#         self.conv1 = nn.Conv2d(1, 64, [2,2], stride=1, padding=0, dilation=1)
#         self.conv2 = nn.Conv2d(64, 64, 1, stride=1, padding=0, dilation=1)
#         self.conv3 = nn.Conv2d(64, 64, 1, stride=1, padding=0, dilation=1)
#         self.conv4 = nn.Conv2d(64, 64, 1, stride=1, padding=0, dilation=1)
#         self.conv5 = nn.Conv2d(64, 64, 1, stride=1, padding=0, dilation=1)
#         self.conv6 = nn.Conv2d(64, num_classes, 1, stride=1, padding=0, dilation=1)
#         #self.pixel_shuffle = nn.PixelShuffle(2)

#         # Init weights
#         for m in self.modules():
#             classname = m.__class__.__name__
#             if classname.lower().find('conv') != -1:
#                 nn.init.kaiming_normal_(m.weight)
#                 nn.init.constant_(m.bias, 0)
#             elif classname.find('bn') != -1:
#                 m.weight.data.normal_(1.0, 0.02)
#                 m.bias.data.fill_(0)


#     def forward(self, x_in):
#         x = self.conv1(x_in)
#         x = self.conv2(F.relu(x))
#         x = self.conv3(F.relu(x))
#         x = self.conv4(F.relu(x))
#         x = self.conv5(F.relu(x))
#         x = self.conv6(F.relu(x))
#         x = torch.sigmoid(x)
#         return x

### A lightweight deep network ###
class SegNet2(torch.nn.Module):
    def __init__(self):
        super(SegNet2, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, [2,2], stride=1, padding=0, dilation=1)
        self.reflect_padding1 = CustomReflectionPad2d(1)
        self.conv2 = nn.Conv2d(64, 64, [2,2], stride=1, padding=0, dilation=2)
        self.reflect_padding2 = CustomReflectionPad2d(1)
        self.conv3 = nn.Conv2d(64, 64, [2,2], stride=1, padding=0, dilation=2)
        self.reflect_padding3 = CustomReflectionPad2d(1)
        self.conv4 = nn.Conv2d(64, 64, [2,2], stride=1, padding=0, dilation=2)
        self.reflect_padding4 = CustomReflectionPad2d(1)
        self.conv5 = nn.Conv2d(64, 64, [2,2], stride=1, padding=0, dilation=2)
        self.reflect_padding5 = CustomReflectionPad2d(1)
        self.conv6 = nn.Conv2d(64, 1, [2,2], stride=1, padding=0, dilation=2)


        # Init weights
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.lower().find('conv') != -1:
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif classname.find('bn') != -1:
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)


    def forward(self, x_in):
        x = self.conv1(x_in)
        x = self.reflect_padding1(x)
        x = self.conv2(F.relu(x))
        x = self.reflect_padding2(x)
        x = self.conv3(F.relu(x))
        x = self.reflect_padding3(x)   
        x = self.conv4(F.relu(x))
        x = self.reflect_padding4(x)
        x = self.conv5(F.relu(x))
        x = self.reflect_padding5(x)
        x = self.conv6(F.relu(x))
        x = torch.sigmoid(x)
        return x

if not isdir('logger'):
    mkdir('logger')
if not isdir('logger/{}'.format(str(VERSION))):
    mkdir('logger/{}'.format(str(VERSION)))
sys.stdout = Print_Logger(os.path.join('logger/{}'.format(str(VERSION)), 'log.txt')) 
print(f"EXP_NAME = {EXP_NAME}")
print(f"VERSION = {VERSION}")
print(f"NB_BATCH = {NB_BATCH}")
print(f"CROP_SIZE = {CROP_SIZE}")
print(f"START_ITER = {START_ITER}")
print(f"NB_ITER = {NB_ITER}")
print(f"I_DISPLAY = {I_DISPLAY}")
print(f"I_VALIDATION = {I_VALIDATION}")
print(f"I_SAVE = {I_SAVE}")
print(f"TRAIN_DIR = {TRAIN_DIR}")
print(f"VAL_DIR = {VAL_DIR}")
print(f"TEST_DIR = {TEST_DIR}")
print(f"LR_G = {LR_G}")
print(f"TEST = {str(TEST)}")

model_G = SegNet2().cuda()

loss_fun = nn.BCELoss().cuda()
print(model_G) 

total_params = sum(p.numel() for p in model_G.parameters())
print(f"Total parameters: {total_params}")

## Optimizers
params_G = list(filter(lambda p: p.requires_grad, model_G.parameters()))
opt_G = optim.Adam(params_G, lr=LR_G)
## Load saved params
if START_ITER > 0:
    lm = torch.load('checkpoint/{}/model_G_i{:06d}.pth'.format(str(VERSION), START_ITER))
    model_G.load_state_dict(lm.state_dict(), strict=True)

    lm = torch.load('checkpoint/{}/opt_G_i{:06d}.pth'.format(str(VERSION), START_ITER))
    opt_G.load_state_dict(lm.state_dict())

if ONLYTEST==False:
    Iter_H = GeneratorEnqueuer(DirectoryIterator_DIV2K( 
                                    datadir = TRAIN_DIR,
                                    crop_size = CROP_SIZE, 
                                    crop_per_image = NB_BATCH//4,
                                    out_batch_size = NB_BATCH,
                                    #scale_factor = UPSCALE,
                                    shuffle=True,
                                    infinite=True))
    Iter_H.start(max_q_size=16, workers=4)

## Prepare directories
if not isdir('checkpoint'):
    mkdir('checkpoint')
if not isdir('result'):
    mkdir('result')
if not isdir('figure'):
    mkdir('figure')
if not isdir('checkpoint/{}'.format(str(VERSION))):
    mkdir('checkpoint/{}'.format(str(VERSION)))
if not isdir('result/{}'.format(str(VERSION))):
    mkdir('result/{}'.format(str(VERSION)))
if not isdir('figure/{}'.format(str(VERSION))):
    mkdir('figure/{}'.format(str(VERSION)))
if not isdir('result/{}'.format(str('test'))):
    mkdir('result/{}'.format(str('test')))


## Some preparations 
print('===> Training start')
l_accum = [0.,0.,0.]
dT = 0.
rT = 0.
accum_samples = 0


def SaveCheckpoint(i, best=False):
    str_best = ''
    if best:
        str_best = '_best'
    torch.save(model_G, 'checkpoint/{}/model_G_i{:06d}{}.pth'.format(str(VERSION), i, str_best ))
    torch.save(opt_G, 'checkpoint/{}/opt_G_i{:06d}{}.pth'.format(str(VERSION), i, str_best))
    print("Checkpoint saved")

best_dice = 0
count_dice =0

### TRAINING
fig_losses = []
fig_num = []
fig_num_val = []
fig_dice = []

if ONLYTEST==False:
    print('\033[0;33m Start to Train\033[0m')
    for i in tqdm(range(START_ITER+1, NB_ITER+1)):
        model_G.train()
        
        # Data preparing
        st = time.time()
        batch_L, batch_H = Iter_H.dequeue()
        batch_H = Variable(torch.from_numpy(batch_H)).cuda()      # BxCxHxW, range [0,1]
        batch_L = Variable(torch.from_numpy(batch_L)).cuda()      # BxCxHxW, range [0,1]
        dT += time.time() - st
        ## TRAIN G
        st = time.time()
        opt_G.zero_grad()
        batch_S = model_G(F.pad(batch_L, (0,1,0,1), mode='reflect'))
    
        loss_G = loss_fun(batch_S, batch_H)
       # Update
        loss_G.backward()
        opt_G.step()
        rT += time.time() - st
  
    # For monitoring
        accum_samples += NB_BATCH
    #l_accum[0] += loss_Contrast.item()
        l_accum[0] += loss_G.item()
        
        ## Show information
        if i % I_DISPLAY == 0:
            fig_num.append((i / I_DISPLAY))
            writer.add_scalar('loss', l_accum[0]/I_DISPLAY, i)
            fig_losses.append(l_accum[0]/I_DISPLAY)
            if USE_WANDB:
                wandb.log({"iteration": i, "loss": l_accum[0]/I_DISPLAY})

            print("{} {}| Iter:{:6d}, Sample:{:6d}, GPixel:{:.2e}, dT:{:.4f}, rT:{:.4f}".format(
                EXP_NAME, VERSION, i, accum_samples, l_accum[0]/I_DISPLAY, dT/I_DISPLAY, rT/I_DISPLAY))
            l_accum = [0.,0.,0.]
            dT = 0.
            rT = 0.

        ## Save models
        if i % I_SAVE == 0:
            SaveCheckpoint(i)

        dice = []
        dice_avg = 0
    
        ## Validation
        if i % I_VALIDATION == 0:
            with torch.no_grad():
                model_G.eval()

                # Test for validation images
                files_gt = glob.glob(VAL_DIR + 'mask/*.png')
                files_gt.sort()
                files_lr = glob.glob(VAL_DIR + 'image/*.png')
                files_lr.sort()
                for ti, fn in enumerate(files_gt):
                    # Load HR image
                    tmp = _load_img_array(files_gt[ti],'BIN')
                    val_H = np.asarray(tmp).astype(np.float32)  # HxWxC
                    # Load LR image
                    tmp_L = _load_img_array(files_lr[ti],'GRAY')
                    val_L = np.asarray(tmp_L).astype(np.float32)  # HxWxC

                  

                    val_L = np.transpose(val_L, [2, 0, 1])      # CxHxW
                    val_L = val_L[np.newaxis, ...]            # BxCxHxW
                    with torch.no_grad():
                        val_L = torch.from_numpy(val_L.copy()).cuda()
                            
                    # Run model
                    batch_S = model_G(F.pad(val_L, (0,1,0,1), mode='reflect'))
                
                    
                    image_out = (batch_S[:,0]).cpu().data.numpy()
            
                    image_out = np.transpose(image_out, [1, 2, 0])  # HxWxC
                    image_out = np.squeeze(image_out, axis=-1)
                    
                    #img_gt = (val_H*255).astype(np.uint8)
                    img_gt = val_H
                    img_gt = np.squeeze(img_gt, axis=-1)
                    image_output=image_out
                    labels, output=add_batch(img_gt,image_out)
                    pred = output>=0.5
                
                    image_output= image_output>=0.5
                    dice.append(f1_score(labels,pred))
                    image_output = ((image_output)*255).astype(np.uint8)
                
                    Image.fromarray(image_output).save('result/{}/{}.png'.format(str(VERSION), fn.split('/')[-1]))
                
                
                dice_avg = np.mean(np.asarray(dice))
            
                dice = []
            
                ## Show information
                print('AVG DICE: Validation: {}'.format(dice_avg))


                if dice_avg > best_dice:
                    best_dice = dice_avg
                    #print('BEST_DICE',best_dice)
                    count_dice = i 
                    SaveCheckpoint(i, best=True)
                print('BEST_DICE',best_dice)
                print('best_iteration',count_dice)
                fig_num_val.append((i / I_VALIDATION))
                fig_dice.append(dice_avg)
                writer.add_scalar('dice', dice_avg, i / I_VALIDATION)
                writer.flush()
                dice_avg = 0
        

    writer.close()
    save_path = 'figure/{}/'.format(str(VERSION))


    plot_dice(fig_num_val, fig_dice, save_path)
    plot_loss(fig_num, fig_losses, save_path)
    if USE_WANDB:
        wandb.finish()

## Validation
if TEST:
    with torch.no_grad():
        best_iteration = count_dice  # 结果最好的轮数
        if ONLYTEST:
            best_model_path = 'checkpoint/{}/model_G_i{}_best.pth'.format(str(VERSION),str(BEST_MODEL))
            print(best_model_path)
        else:
            best_model_path = 'checkpoint/{}/model_G_i{:06d}_best.pth'.format(str(VERSION), best_iteration)
            print(best_model_path)


        model_G = torch.load(best_model_path)
        model_G.eval()
        # Test for validation images
        files_gt = glob.glob(TEST_DIR + 'mask/*.png')
        files_gt.sort()
        files_lr = glob.glob(TEST_DIR + 'image/*.png')
        files_lr.sort()
    

        dice_test = []
        accuracies = []
        sensitivities = []
        specificities = []
 
        #dice_avg_test = 0
        count=0
        for ti, fn in enumerate(files_gt):
            count+=1
            # Load HR image
            tmp = _load_img_array(files_gt[ti],'BIN')
            val_H = np.asarray(tmp).astype(np.float32)  # HxWxC

            # Load LR image
            tmp_L = _load_img_array(files_lr[ti],'GRAY')
            val_L = np.asarray(tmp_L).astype(np.float32)  # HxWxC
            val_L = np.transpose(val_L, [2, 0, 1])      # CxHxW
            val_L = val_L[np.newaxis, ...]            # BxCxHxW
            with torch.no_grad():
                val_L = torch.from_numpy(val_L.copy()).cuda()
            # Output 
            batch_S = model_G(F.pad(val_L, (0,1,0,1), mode='reflect'))
            image_out = (batch_S[:,0]).cpu().data.numpy()
            image_out = np.transpose(image_out, [1, 2, 0])  # HxWxC
            image_out = np.squeeze(image_out, axis=-1)
            
            img_gt = val_H
            img_gt = np.squeeze(img_gt, axis=-1)
            image_output=image_out

            labels, output=add_batch(img_gt,image_out)
            pred = output>=0.5
            dice_test.append(f1_score(labels,pred))

            conf_matrix = confusion_matrix(labels, pred)
            TN, FP, FN, TP = conf_matrix.ravel()
            acc= accuracy_score(labels, pred)
            sens = recall_score(labels, pred)
            spec = TN / (TN + FP)
            accuracies.append(acc)
            sensitivities.append(sens)
            specificities.append(spec)

            image_output = image_output>=0.5
            image_output = ((image_output)*255).astype(np.uint8)
            Image.fromarray(image_output).save('result/{}/{}.png'.format(str('test'), fn.split('/')[-1]))

        print(count)
        dice_avg_test = np.mean(np.asarray(dice_test))
        avg_accuracy = np.mean(accuracies)
        avg_sensitivity = np.mean(sensitivities)
        avg_specificity = np.mean(specificities)
         
        print('TEST AVG dice : {}'.format(dice_avg_test))
        print('TEST AVG Accuracy: {}'.format(avg_accuracy))
        print('TEST AVG Sensitivity: {}'.format(avg_sensitivity))
        print('TESTAVG Specificity: {}'.format(avg_specificity))        
       