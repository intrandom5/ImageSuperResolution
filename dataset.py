import glob
import random
from utils import *
from tqdm import tqdm


class SuperResolutionDataset:
    def __init__(self, directory: str, ratio: int, patch_size: tuple, train: bool):
        self.ratio = ratio
        self.train = train
        self.patch_size = patch_size
        
        print("load images...")
        files = glob.glob(directory + "/*.png")
        self.hr = [load_img(file) for file in tqdm(files)]
        print("Done!")
        
    def shuffle_batch(self):
        random.shuffle(self.hr)
    
    def preprocess_img(self, img, w, h):
        img = pad_img(img, w, h)
        patches = crop_img(img, w, h)
        return patches
    
    def __len__(self):
        return len(self.hr)
    
    def get_batch(self, idx):
        hr = self.hr[idx]
        if self.train:
            hr = aug_img(hr)
        lr = degradation(hr, self.ratio)
        
        orig_w, orig_h = hr.shape[:2]
        w, h = self.patch_size
        
        hr_patches = self.preprocess_img(hr, int(w*self.ratio), int(h*self.ratio))
        lr_patches = self.preprocess_img(lr, w, h)
        
        return {
            "hr": hr_patches/255, 
            "lr": lr_patches/255, 
            "orig_size": (orig_w, orig_h)
        }
