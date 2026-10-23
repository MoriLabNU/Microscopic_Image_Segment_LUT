"""LUT-based segmentation core (4D simplex interpolation; see paper Eq.3 / Table 3)."""
import numpy as np

def four_simplex_interp(weight, img_in, h, w, q, sampling_interval, rot=0, upscale=1):
    L = 2 ** (8 - sampling_interval) + 1

    # Extract MSBs
    img_a1 = img_in[:, 0:0+h, 0:0+w] // q
    
    img_b1 = img_in[:, 0:0+h, 1:1+w] // q
    img_c1 = img_in[:, 1:1+h, 0:0+w] // q
    img_d1 = img_in[:, 1:1+h, 1:1+w] // q
 
    #用于后续求顶点
    img_a2 = img_a1 + 1
    img_b2 = img_b1 + 1
    img_c2 = img_c1 + 1
    img_d2 = img_d1 + 1

    # Extract LSBs
    fa_ = img_in[:, 0:0+h, 0:0+w] % q
    
    fb_ = img_in[:, 0:0+h, 1:1+w] % q
    fc_ = img_in[:, 1:1+h, 0:0+w] % q
    fd_ = img_in[:, 1:1+h, 1:1+w] % q


    # Vertices (O in Eq3 and Table3 in the paper)
    p0000 = weight[ img_a1.flatten().astype(np.int_)*L*L*L + img_b1.flatten().astype(np.int_)*L*L + img_c1.flatten().astype(np.int_)*L + img_d1.flatten().astype(np.int_) ].reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2]))
    p0001 = weight[ img_a1.flatten().astype(np.int_)*L*L*L + img_b1.flatten().astype(np.int_)*L*L + img_c1.flatten().astype(np.int_)*L + img_d2.flatten().astype(np.int_) ].reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2]))
    p0010 = weight[ img_a1.flatten().astype(np.int_)*L*L*L + img_b1.flatten().astype(np.int_)*L*L + img_c2.flatten().astype(np.int_)*L + img_d1.flatten().astype(np.int_) ].reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2]))
    p0011 = weight[ img_a1.flatten().astype(np.int_)*L*L*L + img_b1.flatten().astype(np.int_)*L*L + img_c2.flatten().astype(np.int_)*L + img_d2.flatten().astype(np.int_) ].reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2]))
    p0100 = weight[ img_a1.flatten().astype(np.int_)*L*L*L + img_b2.flatten().astype(np.int_)*L*L + img_c1.flatten().astype(np.int_)*L + img_d1.flatten().astype(np.int_) ].reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2]))
    p0101 = weight[ img_a1.flatten().astype(np.int_)*L*L*L + img_b2.flatten().astype(np.int_)*L*L + img_c1.flatten().astype(np.int_)*L + img_d2.flatten().astype(np.int_) ].reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2]))
    p0110 = weight[ img_a1.flatten().astype(np.int_)*L*L*L + img_b2.flatten().astype(np.int_)*L*L + img_c2.flatten().astype(np.int_)*L + img_d1.flatten().astype(np.int_) ].reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2]))
    p0111 = weight[ img_a1.flatten().astype(np.int_)*L*L*L + img_b2.flatten().astype(np.int_)*L*L + img_c2.flatten().astype(np.int_)*L + img_d2.flatten().astype(np.int_) ].reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2]))
    
    p1000 = weight[ img_a2.flatten().astype(np.int_)*L*L*L + img_b1.flatten().astype(np.int_)*L*L + img_c1.flatten().astype(np.int_)*L + img_d1.flatten().astype(np.int_) ].reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2]))
    p1001 = weight[ img_a2.flatten().astype(np.int_)*L*L*L + img_b1.flatten().astype(np.int_)*L*L + img_c1.flatten().astype(np.int_)*L + img_d2.flatten().astype(np.int_) ].reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2]))
    p1010 = weight[ img_a2.flatten().astype(np.int_)*L*L*L + img_b1.flatten().astype(np.int_)*L*L + img_c2.flatten().astype(np.int_)*L + img_d1.flatten().astype(np.int_) ].reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2]))
    p1011 = weight[ img_a2.flatten().astype(np.int_)*L*L*L + img_b1.flatten().astype(np.int_)*L*L + img_c2.flatten().astype(np.int_)*L + img_d2.flatten().astype(np.int_) ].reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2]))
    p1100 = weight[ img_a2.flatten().astype(np.int_)*L*L*L + img_b2.flatten().astype(np.int_)*L*L + img_c1.flatten().astype(np.int_)*L + img_d1.flatten().astype(np.int_) ].reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2]))
    p1101 = weight[ img_a2.flatten().astype(np.int_)*L*L*L + img_b2.flatten().astype(np.int_)*L*L + img_c1.flatten().astype(np.int_)*L + img_d2.flatten().astype(np.int_) ].reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2]))
    p1110 = weight[ img_a2.flatten().astype(np.int_)*L*L*L + img_b2.flatten().astype(np.int_)*L*L + img_c2.flatten().astype(np.int_)*L + img_d1.flatten().astype(np.int_) ].reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2]))
    p1111 = weight[ img_a2.flatten().astype(np.int_)*L*L*L + img_b2.flatten().astype(np.int_)*L*L + img_c2.flatten().astype(np.int_)*L + img_d2.flatten().astype(np.int_) ].reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2]))
    
    # Output image holder
    out = np.zeros((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2]))
    q = np.full(img_a1.shape, q)
    q = np.array(q, dtype='float32')

    # Naive pixelwise output value interpolation (Table3 in the paper)
    # It would be faster implemented with a parallel operation
    #print("img_a1",img_a1.shape) #img_a1 (1, 1024, 1024)
   
    cond_a_gt_b = fa_ > fb_
    cond_a_gt_c = fa_ > fc_
    cond_a_gt_d = fa_ > fd_
    cond_b_gt_a = fb_ > fa_
    cond_b_gt_c = fb_ > fc_
    cond_b_gt_d = fb_ > fd_
    cond_c_gt_a = fc_ > fa_
    cond_c_gt_b = fc_ > fb_
    cond_c_gt_d = fc_ > fd_
    cond_d_gt_a = fd_ > fa_
    cond_d_gt_b = fd_ > fb_      
    cond_d_gt_c = fd_ > fc_    

    cond_a_eq_b = fa_ == fb_
    cond_a_eq_c = fa_ == fc_
    cond_a_eq_d = fa_ == fd_
    cond_b_eq_c = fb_ == fc_
    cond_b_eq_d = fb_ == fd_
    cond_c_eq_d = fc_ == fd_



    #a>b>c>d
    a1 = (((cond_a_gt_b) & cond_b_gt_c) & cond_c_gt_d)
    out += (((cond_a_gt_b) & cond_b_gt_c) & cond_c_gt_d) * ((q-fa_) * p0000 + (fa_-fb_) * p1000 + (fb_-fc_) * p1100 + (fc_-fd_) * p1110 + fd_ * p1111)
    #a>b>d>c
    a2 = (((cond_a_gt_b) & cond_b_gt_c) & ((~cond_c_gt_d) & cond_b_gt_d))
    out += (((cond_a_gt_b) & cond_b_gt_c) & ((~cond_c_gt_d) & cond_b_gt_d)) * ((q-fa_) * p0000 + (fa_-fb_) * p1000 + (fb_-fd_) * p1100 + (fd_-fc_) * p1101 + fc_ * p1111)
    #a>d>b>c
    a3 = (((cond_a_gt_b) & cond_b_gt_c) & ((~cond_c_gt_d)&(~cond_b_gt_d)&(cond_a_gt_d)))
    out += (((cond_a_gt_b) & cond_b_gt_c) & ((~cond_c_gt_d)&(~cond_b_gt_d)&(cond_a_gt_d)))* ((q-fa_) * p0000 + (fa_-fd_) * p1000 + (fd_-fb_) * p1001 + (fb_-fc_) * p1101 + fc_ * p1111)
    #d>a>b>c
    a4 = (((cond_a_gt_b) & cond_b_gt_c) & ((~cond_c_gt_d)&(~cond_b_gt_d)&(~cond_a_gt_d)))
    out += (((cond_a_gt_b) & cond_b_gt_c) & ((~cond_c_gt_d)&(~cond_b_gt_d)&(~cond_a_gt_d))) * ((q-fd_) * p0000 + (fd_-fa_) * p0001 + (fa_-fb_) * p1001 + (fb_-fc_) * p1101 + fc_ * p1111)

    #a>c>b>d
    a5 = (((cond_a_gt_b) & ((~cond_b_gt_c) & cond_a_gt_c)) & cond_b_gt_d)
    out += (((cond_a_gt_b) & ((~cond_b_gt_c) & cond_a_gt_c)) & cond_b_gt_d) * ((q-fa_) * p0000 + (fa_-fc_) * p1000 + (fc_-fb_) * p1010 + (fb_-fd_) * p1110 + fd_ * p1111)
    #a>c>d>b
    a6 = (((cond_a_gt_b) & ((~cond_b_gt_c) & cond_a_gt_c)) & ((~cond_b_gt_d)&(cond_c_gt_d)))
    out += (((cond_a_gt_b) & ((~cond_b_gt_c) & cond_a_gt_c)) & ((~cond_b_gt_d)&(cond_c_gt_d))) * ((q-fa_) * p0000 + (fa_-fc_) * p1000 + (fc_-fd_) * p1010 + (fd_-fb_) * p1011 + fb_ * p1111)
    #a>d>c>b
    a7 = (((cond_a_gt_b) & ((~cond_b_gt_c) & cond_a_gt_c)) & ((~cond_b_gt_d)&(~cond_c_gt_d)&(cond_a_gt_d)))
    out += (((cond_a_gt_b) & ((~cond_b_gt_c) & cond_a_gt_c)) & ((~cond_b_gt_d)&(~cond_c_gt_d)&(cond_a_gt_d))) * ((q-fa_) * p0000 + (fa_-fd_) * p1000 + (fd_-fc_) * p1001 + (fc_-fb_) * p1011 + fb_ * p1111)
    #d>a>c>b
    a8 = (((cond_a_gt_b) & ((~cond_b_gt_c) & cond_a_gt_c)) & ((~cond_b_gt_d)&(~cond_c_gt_d)&(~cond_a_gt_d)))
    out += (((cond_a_gt_b) & ((~cond_b_gt_c) & cond_a_gt_c)) & ((~cond_b_gt_d)&(~cond_c_gt_d)&(~cond_a_gt_d))) * ((q-fd_) * p0000 + (fd_-fa_) * p0001 + (fa_-fc_) * p1001 + (fc_-fb_) * p1011 + fb_ * p1111)

    #c>a>b>d
    a9 = (((cond_a_gt_b) & ((~cond_b_gt_c)&(~cond_a_gt_c))) & cond_b_gt_d)
    out += (((cond_a_gt_b) & ((~cond_b_gt_c)&(~cond_a_gt_c))) & cond_b_gt_d) * ((q-fc_) * p0000 + (fc_-fa_) * p0010 + (fa_-fb_) * p1010 + (fb_-fd_) * p1110 + fd_ * p1111)
    #c>a>d>b
    a10 = (((cond_a_gt_b) & ((~cond_b_gt_c)&(~cond_a_gt_c))) & ((~cond_b_gt_d)&(cond_c_gt_d)))
    out += (((cond_a_gt_b) & ((~cond_b_gt_c)&(~cond_a_gt_c))) & ((~cond_b_gt_d)&(cond_c_gt_d))) * ((q-fc_) * p0000 + (fc_-fa_) * p0010 + (fa_-fd_) * p1010 + (fd_-fb_) * p1011 + fb_ * p1111)
    #c>d>a>b
    a11 = (((cond_a_gt_b) & ((~cond_b_gt_c)&(~cond_a_gt_c))) & ((~cond_b_gt_d)&(~cond_c_gt_d)&(cond_a_gt_d)))
    out += (((cond_a_gt_b) & ((~cond_b_gt_c)&(~cond_a_gt_c))) & ((~cond_b_gt_d)&(~cond_c_gt_d)&(cond_a_gt_d))) * ((q-fc_) * p0000 + (fc_-fd_) * p0010 + (fd_-fa_) * p0011 + (fa_-fb_) * p1011 + fb_ * p1111)
    #d>c>a>b
    a12 = (((cond_a_gt_b) & ((~cond_b_gt_c)&(~cond_a_gt_c))) & ((~cond_b_gt_d)&(~cond_c_gt_d)&(~cond_a_gt_d)))
    out += (((cond_a_gt_b) & ((~cond_b_gt_c)&(~cond_a_gt_c))) & ((~cond_b_gt_d)&(~cond_c_gt_d)&(~cond_a_gt_d))) * ((q-fd_) * p0000 + (fd_-fc_) * p0001 + (fc_-fa_) * p0011 + (fa_-fb_) * p1011 + fb_ * p1111)
###########################
    #b>a>c>d
    a13 = (((~cond_a_gt_b) & cond_a_gt_c) & cond_c_gt_d)
    out += (((~cond_a_gt_b) & cond_a_gt_c) & cond_c_gt_d) * ((q-fb_) * p0000 + (fb_-fa_) * p0100 + (fa_-fc_) * p1100 + (fc_-fd_) * p1110 + fd_ * p1111)
    #b>a>d>c
    a14 = (((~cond_a_gt_b) & cond_a_gt_c) & ((~cond_c_gt_d)&(cond_a_gt_d)))
    out += (((~cond_a_gt_b) & cond_a_gt_c) & ((~cond_c_gt_d)&(cond_a_gt_d))) * ((q-fb_) * p0000 + (fb_-fa_) * p0100 + (fa_-fd_) * p1100 + (fd_-fc_) * p1101 + fc_ * p1111)
    #b>d>a>c
    a15 = (((~cond_a_gt_b) & cond_a_gt_c) & ((~cond_c_gt_d)&(~cond_a_gt_d)&(cond_b_gt_d)))
    out += (((~cond_a_gt_b) & cond_a_gt_c) & ((~cond_c_gt_d)&(~cond_a_gt_d)&(cond_b_gt_d))) * ((q-fb_) * p0000 + (fb_-fd_) * p0100 + (fd_-fa_) * p0101 + (fa_-fc_) * p1101 + fc_ * p1111)
    #d>b>a>c
    a16 = (((~cond_a_gt_b) & cond_a_gt_c) & ((~cond_c_gt_d)&(~cond_a_gt_d)&(~cond_b_gt_d)))
    out += (((~cond_a_gt_b) & cond_a_gt_c) & ((~cond_c_gt_d)&(~cond_a_gt_d)&(~cond_b_gt_d))) * ((q-fd_) * p0000 + (fd_-fb_) * p0001 + (fb_-fa_) * p0101 + (fa_-fc_) * p1101 + fc_ * p1111)

    #b>c>a>d
    a17 = (((~cond_a_gt_b) & ((~cond_a_gt_c)&cond_b_gt_c)) & cond_a_gt_d)
    out += (((~cond_a_gt_b) & ((~cond_a_gt_c)&cond_b_gt_c)) & cond_a_gt_d) * ((q-fb_) * p0000 + (fb_-fc_) * p0100 + (fc_-fa_) * p0110 + (fa_-fd_) * p1110 + fd_ * p1111)
    #b>c>d>a
    a18 = (((~cond_a_gt_b) & ((~cond_a_gt_c)&cond_b_gt_c)) & ((~cond_a_gt_d)&(cond_c_gt_d)))
    out += (((~cond_a_gt_b) & ((~cond_a_gt_c)&cond_b_gt_c)) & ((~cond_a_gt_d)&(cond_c_gt_d))) * ((q-fb_) * p0000 + (fb_-fc_) * p0100 + (fc_-fd_) * p0110 + (fd_-fa_) * p0111 + fa_ * p1111)
    #b>d>c>a
    a19 = (((~cond_a_gt_b) & ((~cond_a_gt_c)&cond_b_gt_c)) & ((~cond_a_gt_d)&(~cond_c_gt_d)&(cond_b_gt_d)))
    out += (((~cond_a_gt_b) & ((~cond_a_gt_c)&cond_b_gt_c)) & ((~cond_a_gt_d)&(~cond_c_gt_d)&(cond_b_gt_d))) * ((q-fb_) * p0000 + (fb_-fd_) * p0100 + (fd_-fc_) * p0101 + (fc_-fa_) * p0111 + fa_ * p1111)
    #d>b>c>a
    a20 = (((~cond_a_gt_b) & ((~cond_a_gt_c)&cond_b_gt_c)) & ((~cond_a_gt_d)&(~cond_c_gt_d)&(~cond_b_gt_d)))
    out += (((~cond_a_gt_b) & ((~cond_a_gt_c)&cond_b_gt_c)) & ((~cond_a_gt_d)&(~cond_c_gt_d)&(~cond_b_gt_d))) * ((q-fd_) * p0000 + (fd_-fb_) * p0001 + (fb_-fc_) * p0101 + (fc_-fa_) * p0111 + fa_ * p1111)

    #c>b>a>d
    a21 = (((~cond_a_gt_b) & ((~cond_a_gt_c)&(~cond_b_gt_c))) & cond_a_gt_d)
    out += (((~cond_a_gt_b) & ((~cond_a_gt_c)&(~cond_b_gt_c))) & cond_a_gt_d) * ((q-fc_) * p0000 + (fc_-fb_) * p0010 + (fb_-fa_) * p0110 + (fa_-fd_) * p1110 + fd_ * p1111)
    #c>b>d>a
    a22 = (((~cond_a_gt_b) & ((~cond_a_gt_c)&(~cond_b_gt_c))) & ((~cond_a_gt_d)&(cond_b_gt_d))) 
    out += (((~cond_a_gt_b) & ((~cond_a_gt_c)&(~cond_b_gt_c))) & ((~cond_a_gt_d)&(cond_b_gt_d)))  * ((q-fc_) * p0000 + (fc_-fb_) * p0010 + (fb_-fd_) * p0110 + (fd_-fa_) * p0111 + fa_ * p1111)
    #c>d>b>a
    a23 = (((~cond_a_gt_b) & ((~cond_a_gt_c)&(~cond_b_gt_c))) & ((~cond_a_gt_d)&(~cond_b_gt_d)&(cond_c_gt_d))) 
    out += (((~cond_a_gt_b) & ((~cond_a_gt_c)&(~cond_b_gt_c))) & ((~cond_a_gt_d)&(~cond_b_gt_d)&(cond_c_gt_d)))  * ((q-fc_) * p0000 + (fc_-fd_) * p0010 + (fd_-fb_) * p0011 + (fb_-fa_) * p0111 + fa_ * p1111)
    #d>c>b>a
    a24 = (((~cond_a_gt_b) & ((~cond_a_gt_c)&(~cond_b_gt_c))) & ((~cond_a_gt_d)&(~cond_b_gt_d)&(~cond_c_gt_d))) 
    out += (((~cond_a_gt_b) & ((~cond_a_gt_c)&(~cond_b_gt_c))) & ((~cond_a_gt_d)&(~cond_b_gt_d)&(~cond_c_gt_d))) * ((q-fd_) * p0000 + (fd_-fc_) * p0001 + (fc_-fb_) * p0011 + (fb_-fa_) * p0111 + fa_ * p1111) 

    
    out = out/ q
    return out



def apply_lut_segmentation(img_lr, lut, sampling_interval, upscale=1):
    """img_lr: float32 (H, W, 1). Returns float32 probability map (H, W)."""
    h, w, _ = img_lr.shape
    q = 2 ** sampling_interval
    lut = np.asarray(lut, dtype=np.float32).reshape(-1, upscale * upscale)
    img_in = np.pad(img_lr, ((0, 1), (0, 1), (0, 0)), mode="reflect").transpose((2, 0, 1))
    img_out = four_simplex_interp(lut, img_in, h, w, q, sampling_interval, rot=0, upscale=upscale)
    img_out = (img_out / 255.0).transpose((1, 2, 0))
    return np.squeeze(img_out, axis=-1)


def load_lut(lut_path, upscale=1):
    return np.load(lut_path).astype(np.float32).reshape(-1, upscale * upscale)
