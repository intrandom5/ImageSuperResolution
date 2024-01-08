from utils import *
from models import get_model
from dataset import SuperResolutionDataset

import os
import yaml
import math
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter


def main(config):
    train_dataset = SuperResolutionDataset(
        config["train_dir"], ratio=config["ratio"], patch_size=config["patch_size"], train=True
        )
    valid_dataset = SuperResolutionDataset(
        config["valid_dir"], ratio=config["ratio"], patch_size=config["patch_size"], train=False
        )
    model = get_model(D=config["D"], C=config["C"], G=config["G"], ratio=config["ratio"], verbose=True)

    criterion = nn.L1Loss()
    optimizer = Adam(model.parameters(), lr=config["lr"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"curretn device is on {device}")
    model.to(device)

    writer = SummaryWriter(config["log_dir"])

    for epoch in range(1, config["epochs"]+1):
        print(f"{epoch}/{config['epochs']} epoch training...")
        model.train()
        for step in tqdm(range(1000)):
            i = random.choice(range(0, len(train_dataset)))
            batch = train_dataset.get_batch(i)
            hr = torch.tensor(batch["hr"]).view(-1, 3, config["patch_size"][0]*config["ratio"], config["patch_size"][1]*config["ratio"])
            lr = torch.tensor(batch["lr"]).view(-1, 3, config["patch_size"][0], config["patch_size"][1])

            idx = random.sample(range(hr.shape[0]), config["batch_size"])
            lr_batch = lr[idx, :, :, :].to(device)
            hr_batch = hr[idx, :, :, :].to(device)

            pred = model(lr_batch.float())
            loss = criterion(pred, hr_batch)
            writer.add_scalar("train loss", loss, step+(epoch-1)*1000)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        optimizer.param_groups[0]['lr'] /= 2
        print("learning rate scheduled to", optimizer.param_groups[0]['lr'])

        if epoch % 10 == 0 and epoch > 1:
            model.eval()
            eval_loss = 0
            eval_psnr = 0
            print("evaluate...")
            for i in range(len(valid_dataset)):
                batch = valid_dataset.get_batch(i)
                x, y = batch["lr"].shape[:2]
                hr = torch.tensor(batch["hr"]).view(-1, 3, config["patch_size"][0]*config["ratio"], config["patch_size"][1]*config["ratio"])
                lr = torch.tensor(batch["lr"]).view(-1, 3, config["patch_size"][0], config["patch_size"][1])

                preds = []
                loss = 0
                for j in range(math.ceil(lr.shape[0]/config["batch_size"])):
                    max_range = min(config["batch_size"]*(j+1), lr.shape[0])
                    lr_batch = lr[config["batch_size"]*j:max_range, :, :, :].to(device)
                    hr_batch = hr[config["batch_size"]*j:max_range, :, :, :].to(device)
                    with torch.no_grad():
                        pred = model(lr_batch.float())

                    loss = criterion(pred, hr_batch)
                    loss += loss.cpu()
                    preds.append(pred.cpu())

                    del lr_batch
                    del hr_batch

                eval_loss += loss/(j+1)
                preds = torch.cat(preds, dim=0)
                preds = preds.view(x, y, config["patch_size"][0]*config["ratio"], config["patch_size"][1]*config["ratio"], 3)
                pred_img = collapse_img(preds.numpy())
                pred_img = unpad_img(pred_img, batch["orig_size"][0], batch["orig_size"][1])
                orig_img = collapse_img(batch["hr"])
                orig_img = unpad_img(orig_img, batch["orig_size"][0], batch["orig_size"][1])

                psnr = psnr_metrics(orig_img, pred_img)
                eval_psnr += psnr
                save_path = os.path.join(config["result_path"], f"epochs{epoch}")
                if not os.path.exists(save_path):
                    os.mkdir(save_path)
                save_img(pred_img*255, os.path.join(save_path, f"pred_{i}.png"))
            writer.add_scalar("dev loss", eval_loss/len(valid_dataset), epoch)
            writer.add_scalar("dev psnr", eval_psnr/len(valid_dataset), epoch)
            path = os.path.join(config["model_path"], f"epochs{epoch}.pt")
            torch.save(model.state_dict(), path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='training config')
    parser.add_argument("--conf", type=str, default="configs/config.yaml", help="config file path(.yaml)")
    args = parser.parse_args()

    with open(args.conf, "r") as f:
        config = yaml.load(f, Loader=yaml.Loader)
    config["patch_size"] = eval(config["patch_size"])
    
    main(config)
