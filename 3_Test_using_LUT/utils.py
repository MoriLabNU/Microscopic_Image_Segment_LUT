import torch
import torch.nn as nn
import time
import numpy as np
import threading
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import random

# def set_seed(seed):
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
#     np.random.seed(seed)
#     random.seed(seed)
#     #torch.use_deterministic_algorithms(True)
# set_seed(42)
import torch
import torch.nn as nn


def to_one_hot(inputs, n_classes):
    n, _,h, w = inputs.size()
    one_hot = torch.zeros((n, n_classes, h, w), device=inputs.device).scatter_(1, inputs.view(n, 1, h, w).long(), 1)
    return one_hot
def add_batch(batch_tar,batch_out):
    batch_tar = batch_tar.flatten()
    batch_out = batch_out.flatten()

    return batch_tar,batch_out
class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, input, target):
        N = target.size(0)
        smooth = 1
        #target = to_one_hot(target,2)
        
        input_flat = input.view(N, -1)
        target_flat = target.view(N, -1)

        intersection = input_flat * target_flat

        loss = 2 * (intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
        loss = 1 - loss.sum() / N

        return loss

### COMMON FUNCTIONS ###    
def _rgb2ycbcr(img, maxVal=255):
#    r = img[:,:,0]
#    g = img[:,:,1]
#    b = img[:,:,2]

    O = np.array([[16],
                  [128],
                  [128]])
    T = np.array([[0.256788235294118, 0.504129411764706, 0.097905882352941],
                  [-0.148223529411765, -0.290992156862745, 0.439215686274510],
                  [0.439215686274510, -0.367788235294118, -0.071427450980392]])

#    ycbcr = np.empty([img.shape[0], img.shape[1], img.shape[2]])

    if maxVal == 1:
        O = O / 255.0

#    ycbcr[:,:,0] = ((T[0,0] * r) + (T[0,1] * g) + (T[0,2] * b) + O[0])
#    ycbcr[:,:,1] = ((T[1,0] * r) + (T[1,1] * g) + (T[1,2] * b) + O[1])
#    ycbcr[:,:,2] = ((T[2,0] * r) + (T[2,1] * g) + (T[2,2] * b) + O[2])

    t = np.reshape(img, (img.shape[0]*img.shape[1], img.shape[2]))
    t = np.dot(t, np.transpose(T))
    t[:, 0] += O[0]
    t[:, 1] += O[1]
    t[:, 2] += O[2]
    ycbcr = np.reshape(t, [img.shape[0], img.shape[1], img.shape[2]])

#    print(np.all((ycbcr - ycbcr_) < 1/255.0/2.0))

    return ycbcr


def _load_img_array(path, color_mode='RGB', channel_mean=None, modcrop=[0,0,0,0]):
    '''Load an image using PIL and convert it into specified color space,
    and return it as an numpy array.
    '''
    ## Load image
    from PIL import Image
    img = Image.open(path)
    if color_mode == 'RGB':
        cimg = img.convert('RGB')
        x = np.asarray(cimg, dtype='float32')

    elif color_mode == 'YCbCr' or color_mode == 'Y':
        cimg = img.convert('YCbCr')
        x = np.asarray(cimg, dtype='float32')
        if color_mode == 'Y':
            x = x[:,:,0:1]

    elif color_mode == 'GRAY':
        cimg = img.convert('L')
        x = np.asarray(cimg, dtype='float32')
        x = x[:,:,np.newaxis]  # add a channel dimension
    elif color_mode == 'BIN':
        cimg = img.convert('L')
        x = np.asarray(cimg, dtype='float32')
        x = x[:,:,np.newaxis]  # add a channel dimension

    ## To 0-1
    # if color_mode == 'BIN':
    #     print(x.shape)
    #     x *= 1/255
    #     x = x.astype(np.int64)
    # else:
    x *= 1.0/255.0

    if channel_mean:
        x[:,:,0] -= channel_mean[0]
        x[:,:,1] -= channel_mean[1]
        x[:,:,2] -= channel_mean[2]

    if modcrop[0]*modcrop[1]*modcrop[2]*modcrop[3]:
        x = x[modcrop[0]:-modcrop[1], modcrop[2]:-modcrop[3], :]
    
    # if color_mode == 'BIN':
        
    #     x=x.astype(np.int64)

    return x

    
def PSNR(y_true, y_pred, shave_border=4):
    '''
        Input must be 0-255, 2D
    '''

    target_data = np.array(y_true, dtype=np.float32)
    ref_data = np.array(y_pred, dtype=np.float32)

    diff = ref_data - target_data
    if shave_border > 0:
        diff = diff[shave_border:-shave_border, shave_border:-shave_border]
    rmse = np.sqrt(np.mean(np.power(diff, 2)))

    return 20 * np.log10(255./rmse)




def SSIM(img1, img2, max_val=255, multichannel=False):
    if not img1.shape == img2.shape:
        raise ValueError("Input images must have the same dimensions.")
    if not img1.dtype == img2.dtype:
        raise ValueError("Input images must have the same dtype.")

    return ssim(img1, img2, multichannel=multichannel, data_range=max_val - 0)

# class FMD_Dataset(Dataset):
#     def __init__(self, datadir, crop_size, nb_batch, transform=None):
#         super(FMD_Dataset, self).__init__()

#         self.crop_size = crop_size
#         self.nb_batch = nb_batch
#         self.transform = transform
#         self.datadir = datadir

#         # Get file paths
#         self.noisy_imgs = sorted(os.listdir(os.path.join(self.datadir, 'noisy')))
#         self.clean_imgs = sorted(os.listdir(os.path.join(self.datadir, 'clean')))

#         assert len(self.noisy_imgs) == len(self.clean_imgs), "Noisy and clean images counts do not match"

#     def __len__(self):
#         return len(self.noisy_imgs)
    
#     def __getitem__(self, idx):
#         noisy_img_path = os.path.join(self.datadir, 'noisy', self.noisy_imgs[idx])
#         clean_img_path = os.path.join(self.datadir, 'clean', self.clean_imgs[idx])

#         # Load images
#         noisy_img = Image.open(noisy_img_path).convert('L')
#         clean_img = Image.open(clean_img_path).convert('L')
      

#         # Random crop
#         i, j, h, w = transforms.RandomCrop.get_params(noisy_img, output_size=(self.crop_size, self.crop_size))
#         noisy_img = transforms.functional.crop(noisy_img, i, j, h, w)
#         clean_img = transforms.functional.crop(clean_img, i, j, h, w)

#         # Random horizontal flip
#         if np.random.random() < 0.5:
#             noisy_img = transforms.functional.hflip(noisy_img)
#             clean_img = transforms.functional.hflip(clean_img)

#         # Apply transformations
#         if self.transform is not None:
#             noisy_img = self.transform(noisy_img)
#             clean_img = self.transform(clean_img)
       
#         return noisy_img, clean_img


### DATASET HANDLING ###    
class GeneratorEnqueuer(object):
    """Builds a queue out of a data generator.
    Used in `fit_generator`, `evaluate_generator`, `predict_generator`.
    # Arguments
        generator: a generator function which endlessly yields data
        pickle_safe: use multiprocessing if True, otherwise threading

    **copied from https://github.com/fchollet/keras/blob/master/keras/engine/training.py

    Usage:
    enqueuer = GeneratorEnqueuer(generator, pickle_safe=pickle_safe)
    enqueuer.start(max_q_size=max_q_size, workers=workers)

    while enqueuer.is_running():
        if not enqueuer.queue.empty():
            generator_output = enqueuer.queue.get()
            break
        else:
            time.sleep(wait_time)
    """

    def __init__(self, generator, use_multiprocessing=True, wait_time=0.00001, random_seed=int(time.time())):
        self.wait_time = wait_time
        self._generator = generator
        self._use_multiprocessing = use_multiprocessing
        self._threads = []
        self._stop_event = None
        self.queue = None
        self.random_seed = random_seed



    def start(self, workers=1, max_q_size=10):
        """Kicks off threads which add data from the generator into the queue.
        # Arguments
            workers: number of worker threads
            max_q_size: queue size (when full, threads could block on put())
            wait_time: time to sleep in-between calls to put()
        """

        def data_generator_task():
            while not self._stop_event.is_set():
                try:
                    if self._use_multiprocessing or self.queue.qsize() < max_q_size:
                        generator_output = next(self._generator)
                        self.queue.put(generator_output)
                    else:
                        time.sleep(self.wait_time)
                except Exception:
                    self._stop_event.set()
                    raise

        try:
            import multiprocessing
            try:
                import queue
            except ImportError:
                import Queue as queue

            if self._use_multiprocessing:
                self.queue = multiprocessing.Queue(maxsize=max_q_size)
                self._stop_event = multiprocessing.Event()
            else:
                self.queue = queue.Queue()
                self._stop_event = threading.Event()

            for _ in range(workers):
                if self._use_multiprocessing:
                    # Reset random seed else all children processes
                    # share the same seed
                    np.random.seed(self.random_seed)
                    thread = multiprocessing.Process(target=data_generator_task)
                    thread.daemon = True
                    if self.random_seed is not None:
                        self.random_seed += 1
                else:
                    thread = threading.Thread(target=data_generator_task)
                self._threads.append(thread)
                thread.start()
        except:
            self.stop()
            raise

    def is_running(self):
        return self._stop_event is not None and not self._stop_event.is_set()

    def stop(self, timeout=None):
        """Stop running threads and wait for them to exit, if necessary.
        Should be called by the same thread which called start().
        # Arguments
            timeout: maximum time to wait on thread.join()
        """
        if self.is_running():
            self._stop_event.set()

        for thread in self._threads:
            if thread.is_alive():
                if self._use_multiprocessing:
                    thread.terminate()
                else:
                    thread.join(timeout)

        if self._use_multiprocessing:
            if self.queue is not None:
                self.queue.close()

        self._threads = []
        self._stop_event = None
        self.queue = None


    def dequeue(self):
        while self.is_running():
            if not self.queue.empty():
                return self.queue.get()
                break
            else:
                time.sleep(self.wait_time)



#################################################################
### Batch Iterators #############################################
#################################################################
class Iterator(object):
    '''
    https://github.com/fchollet/keras/blob/master/keras/preprocessing/image.py
    '''
    def __init__(self, N, batch_size, shuffle, seed, infinite):
        self.N = N
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.batch_index = 0
        self.total_batches_seen = 0
        self.lock = threading.Lock()
        self.index_generator = self._flow_index(N, batch_size, shuffle, seed, infinite)

    def reset(self):
        self.batch_index = 0

    def _flow_index(self, N, batch_size=32, shuffle=False, seed=None, infinite=True):
        # ensure self.batch_index is 0
        self.reset()
        while 1:
            if seed is not None:
                np.random.seed(seed + self.total_batches_seen)
            if self.batch_index == 0:
                index_array = np.arange(N)
                if shuffle:
                    index_array = np.random.permutation(N)

            if infinite == True:
                current_index = (self.batch_index * batch_size) % N
                if N >= current_index + batch_size:
                    current_batch_size = batch_size
                    self.batch_index += 1
                else:
                    current_batch_size = N - current_index
                    self.batch_index = 0
            else:
                current_index = (self.batch_index * batch_size)
                if current_index >= N:
                    self.batch_index = 0
                    raise StopIteration()
                elif N >= current_index + batch_size:
                    current_batch_size = batch_size
                else:
                    current_batch_size = N - current_index
                self.batch_index += 1
            self.total_batches_seen += 1
            yield (index_array[current_index: current_index + current_batch_size],
                   current_index, current_batch_size)

    def __iter__(self):
        # needed if we want to do something like:
        # for x, y in data_gen.flow(...):
        return self

    def __next__(self, *args, **kwargs):
        return self.next(*args, **kwargs)


def _flip_axis(x, axis):
    x = np.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x


class DirectoryIterator_DIV2K(Iterator):
    def __init__(self,
                datadir = './train/',
                crop_size = 32,
                crop_per_image = 4,
                out_batch_size = 16,
                #scale_factor = 4,
                shuffle = True,
                seed = None,
                infinite = True):

        self.crop_size = crop_size
        self.out_batch_size = out_batch_size
        self.crop_per_image = crop_per_image
        self.datadir = datadir
        #self.r = scale_factor
        self.shuffle = shuffle

        if seed is None:
            seed = int(time.time())

        import glob
        lrs = glob.glob(datadir+'/image/*.png')
        lrs.sort()

        sharps = glob.glob(datadir+'/mask/*.png')
        sharps.sort()

        if len(lrs) != len(sharps):
            print("file count mismatch")
            raise

        self.lr_pngs = lrs
        self.sharp_pngs = sharps
        self.total_count = len(lrs)
        self.first_run = True

        print('Found %d images' % self.total_count)

        super(DirectoryIterator_DIV2K, self).__init__(self.total_count, out_batch_size//crop_per_image, shuffle, seed, infinite)


    def shuffle_list(self, *ls):
        import random
        random.seed(time.time())
        l = list(zip(*ls))
        random.shuffle(l)
        return zip(*l)

    def next(self):
        # do shuffle at the very first
        if self.shuffle and self.first_run:
            self.first_run = False
            if self.shuffle:
                self.lr_pngs, self.sharp_pngs = self.shuffle_list(self.lr_pngs, self.sharp_pngs)

        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)
        # The transformation of images is not under thread lock so it can be done in parallel

        batch_blur = []
        batch_sharp = []
        
        i = 0
        while (len(batch_blur) < self.out_batch_size):
            blurs = self.lr_pngs[(current_index+i) % self.total_count]
            sharps = self.sharp_pngs[(current_index+i) % self.total_count]

            # check if png file is damaged
            try:
                B_ = _load_img_array(blurs,color_mode='GRAY') 
                S_ = _load_img_array(sharps,color_mode='BIN') 
            except:
                print("File open error: {} {}".format(blurs, sharps))
                raise

            for j in range(self.crop_per_image):
                if (len(batch_blur) >= self.out_batch_size):
                    break

                bs = B_.shape   # h, w, c
                if bs[0] < self.crop_size or bs[1] < self.crop_size:
                    break
                
                sh = np.random.randint(0, bs[0]-self.crop_size+1)
                sw = np.random.randint(0, bs[1]-self.crop_size+1)
                B = B_[sh:sh+self.crop_size, sw:sw+self.crop_size]
                S = S_[sh:sh+self.crop_size, sw:sw+self.crop_size]

                # Random Aug
                # Rot
                ri = np.random.randint(0,4)
                B = np.rot90(B, ri)
                S = np.rot90(S, ri)

                # LR flip
                if np.random.random() < 0.5:
                    B = _flip_axis(B, 1)
                    S = _flip_axis(S, 1)

                batch_blur.append(B)
                batch_sharp.append(S)

            i += 1
            
        batch_blur = np.stack(batch_blur, 0).astype(np.float32) # BxHxWxC
        batch_sharp = np.stack(batch_sharp, 0).astype(np.float32)

        return batch_blur.transpose((0,3,1,2)), batch_sharp.transpose((0,3,1,2))





def plot_dice(iterations, dice, save_path):
    plt.figure()
    plt.plot(iterations, dice, label='DICE')
    plt.xlabel('Iteration')
    plt.ylabel('DICE')
    plt.legend()
    plt.savefig(f"{save_path}/dice_plot.png")


def plot_ssim(iterations, ssims, save_path):
    plt.figure()
    plt.plot(iterations, ssims, label='SSIM')
    plt.xlabel('Iteration')
    plt.ylabel('SSIM')
    plt.legend()
    plt.savefig(f"{save_path}/ssim_plot.png")


def plot_loss(iterations, losses, save_path):
    plt.figure()
    plt.plot(iterations, losses, label='Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f"{save_path}/loss_plot.png")

