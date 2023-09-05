
import torch
from timm.utils import NativeScaler
from torch.utils.tensorboard import SummaryWriter
import time

import torchmetrics as tm

from torch.utils.data import DataLoader

from model import *
from dataset import *

EPOCHS = 200
BATCH_SIZE = 4

train_dir = "dataset/GoPro/train"
valid_dir = "dataset/GoPro/valid"

def train():
    """
     @brief Train UFormer on data stored in train_dir and validate it on valid_dir.
    """
    train_dataset = get_training_data(train_dir)
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True,
            num_workers=1, pin_memory=True, drop_last=False)
    val_dataset = get_validation_data(valid_dir)
    val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=1, pin_memory=True, drop_last=False)


    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    psnr = tm.PeakSignalNoiseRatio().to(device)
    ssim = tm.StructuralSimilarityIndexMeasure().to(device)

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    loss_scaler = NativeScaler()
    torch.cuda.empty_cache()



    #initialize UFormer with default (paper) parameters
    model = UFormer()
    model.to(device)
    criterion = CharbonnierLoss()
    optimizer = torch.optim.AdamW(model.parameters(),1E-4,weight_decay=0.002)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,EPOCHS,eta_min=1E-6)
    logger = SummaryWriter()
    best_model_psnr=0
    # This is the loop used to train and eval the model.
    for epoch in range(EPOCHS):
        epoch_start_time = time.time()
        train_loss = 0
        valid_loss = 0
        model.train()
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()
            target = data[0].to(device)
            input = data[1].to(device)
            with torch.cuda.amp.autocast():
                pred = model(input)
                loss = criterion(pred,target)
            loss_scaler(loss,optimizer,parameters=model.parameters())
            train_loss+=loss.item()
        logger.add_scalar("Train Loss",train_loss,epoch)
        model.eval()
        with torch.no_grad():
            psnr_scalar = 0
            ssim_scalar = 0
            for i, data in enumerate(val_loader):
                target = data[0].to(device)
                input = data[1].to(device)
                with torch.cuda.amp.autocast():
                    pred = model(input)
                loss = criterion(pred,target)
                valid_loss+=loss.item()
                psnr_scalar+=psnr(pred,target).item()
                ssim_scalar+=ssim(pred,target).item()
            logger.add_scalar("Valid Loss",valid_loss,epoch)
        # Save the model.
        if psnr_scalar>best_model_psnr:
            best_model_psnr=psnr_scalar
            model.cpu()
            torch.save(model.state_dict(),f"checkpoints/ckpt_{epoch}_{best_model_psnr:.2f}.pt")
            model.to(device)
        logger.add_scalar("PSNR",psnr_scalar/(i+1),epoch)
        logger.add_scalar("SSIM",ssim_scalar/(i+1),epoch)
        stop = time.time() - epoch_start_time
        print("@EPOCH {}: Train Loss {:.3f}; Valid Loss {:.3f}; PSNR {:.3f}; SSIM {:.3f}; Time {:.3f}".format(epoch,train_loss,valid_loss,psnr_scalar/(i+1),ssim_scalar/(i+1),stop))
        scheduler.step()

if __name__=="__main__":
    train()