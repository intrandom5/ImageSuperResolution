import os
import yaml
import glob
import torch
import argparse
from tqdm import tqdm
from utils import *
from models import get_model


def main(config):
    # load img files
    path = config["img_path"]
    if path.split(".")[-1] in ["jpg", "png", "jpeg"]:
        imgs = [load_img(path)]
        file_names = [path.split("/")[-1]]
    else:
        files = glob.glob(path+"/*")
        imgs = []
        file_names = []
        for file in files:
            if file.split(".")[-1] in ["jpg", "png", "jpeg"]:
                imgs.append(file)
                file_names.append(file.split("\\")[-1])
        imgs = [load_img(img) for img in imgs]
    print(f"found {len(imgs)} image files.")
    
    # load model
    print(f"load model from '{config['model_path']}'")
    model_state = torch.load(config["model_path"])
    model = get_model(D=config["D"], C=config["C"], G=config["G"], ratio=config["ratio"], verbose=False)
    model.load_state_dict(model_state)

    # use cuda
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("inference will be on", device)
    model.to(device)
    model.eval()

    w, h = config["patch_size"]

    # make directory of SR result path.
    if not os.path.exists(config["save_path"]):
        os.mkdir(config["save_path"])
    print(f"result images will be save to '{config['save_path']}'")

    # do SR for imgs
    print("Doing Super Resolution...")
    for z in tqdm(range(len(imgs))):
        img = imgs[z]
        file_name = file_names[z]
        orig_w = img.shape[0]*config["ratio"]
        orig_h = img.shape[1]*config["ratio"]
        img = pad_img(img, w, h)
        patches = crop_img(img, w, h)
        patches /= 255
        x, y = patches.shape[:2]
        patches = torch.tensor(patches).view(-1, 3, w, h)
        preds = []

        for i in range(math.ceil(patches.shape[0]/config["batch_size"])):
            max_range = min(config["batch_size"]*(i+1), patches.shape[0])
            patch_batch = patches[config["batch_size"]*i:max_range].to(device)
            with torch.no_grad():
                pred = model(patch_batch.float().to(device))
            preds.append(pred.cpu())

        preds = torch.cat(preds, dim=0)
        preds = preds.view(x, y, w*config["ratio"], h*config["ratio"], 3)
        preds = collapse_img(preds.numpy())
        preds = unpad_img(preds, orig_w, orig_h)
        dst = os.path.join(config["save_path"], file_name)
        save_img(preds*255, dst)
        print(f"completed. save image to '{dst}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='training config')
    parser.add_argument("--conf", type=str, default="configs/eval_config.yaml", help="config file path(.yaml)")
    args = parser.parse_args()

    with open(args.conf, "r") as f:
        config = yaml.load(f, Loader=yaml.Loader)
    config["patch_size"] = eval(config["patch_size"])
    
    main(config)