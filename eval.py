import torch

import time

import torchmetrics as tm

from torch.utils.data import DataLoader

from model import *
from dataset import *


valid_dir = "dataset/GoPro/valid"

def eval(ckpt_path):
    val_dataset = get_validation_data(valid_dir)
    val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False,
            num_workers=1, pin_memory=True, drop_last=False)


    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    psnr = tm.PeakSignalNoiseRatio().to(device)
    ssim = tm.StructuralSimilarityIndexMeasure().to(device)

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    torch.cuda.empty_cache()



    #initialize UFormer with default (paper) parameters
    model = UFormer()
    model.load_state_dict(torch.load(ckpt_path),strict=False)
    model.to(device)

    model.eval()
    with torch.no_grad():
        psnr_scalar = 0
        ssim_scalar = 0
        for i, data in enumerate(val_loader):
            target = data[0].to(device)
            input = data[1].to(device)
            with torch.cuda.amp.autocast():
                pred = model(input)
            psnr_scalar+=psnr(pred,target).item()
            ssim_scalar+=ssim(pred,target).item()
            print("@EPOCH {}: PSNR {:.3f}; SSIM {:.3f}; Time {:.3f}".format(psnr_scalar/(i+1),ssim_scalar/(i+1),))

if __name__=="__main__":
    weights = "checkpoint/ckpt_192_7010.46.pt"
    eval(weights)