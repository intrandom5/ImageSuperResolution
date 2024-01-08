import cv2
import math
import random
import numpy as np
from PIL import Image


def load_img(file):
    img = cv2.imread(file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def save_img(img, name):
    img = Image.fromarray(np.uint8(img))
    img.save(name)

def degradation(hr, ratio):
    lr = cv2.resize(hr, (hr.shape[1]//ratio, hr.shape[0]//ratio), cv2.INTER_CUBIC)
    return lr

def pad_img(img, w=32, h=32):
    width, height = img.shape[:2]
    padded_width = int(w*(np.ceil(width/w)))
    padded_height = int(h*(np.ceil(height/h)))
    
    padded_img = np.zeros((padded_width, padded_height, 3))
    padded_img[:width, :height, :] = img
    return padded_img

def crop_img(img, w=32, h=32):
    img = pad_img(img, w, h)
    width, height = img.shape[:2]
    w_num = width//w
    h_num = height//h
    
    patches = []
    
    for i in range(w_num):
        patches.append([])
        for j in range(h_num):
            patch = img[i*w:(i+1)*w, j*h:(j+1)*h]
            patches[i].append(patch)
            
    return np.array(patches)

def collapse_img(crops):
    w_num, h_num, w, h, _ = crops.shape
    full_img = np.zeros((w*w_num, h*h_num, 3))
    for w_ in range(w_num):
        for h_ in range(h_num):
            full_img[w_*w:(w_+1)*w, h_*h:(h_+1)*h, :] = crops[w_][h_]
    
    return full_img

def unpad_img(img, orig_w, orig_h):
    return img[:orig_w, :orig_h, :]

def aug_img(img):
    rotate_ops = ["0", "90", "180", "270"]
    flip_ops = ["none", "horizontal", "vertical"]
    rotate = random.choice(rotate_ops)
    flip = random.choice(flip_ops)
    if rotate == "90":
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif rotate == "180":
        img = cv2.rotate(img, cv2.ROTATE_180)
    elif rotate == "270":
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if flip == "horizontal":
        img = cv2.flip(img, 0)
    elif flip == "vertical":
        img = cv2.flip(img, 1)
    
    return img

def psnr_metrics(original, contrast):
    '''
    https://github.com/jackfrued/Python-1/blob/master/analysis/compression_analysis/psnr.py
    '''
    mse = np.mean((original - contrast) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    PSNR = 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

    return PSNR
