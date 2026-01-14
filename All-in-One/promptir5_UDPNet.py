import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import torchmetrics.image
from utils import schedulers
from options import options as opt
from utils.schedulers import LinearWarmupCosineAnnealingLR
from net.promptir_depth import PromptIR
from test import *
import torchmetrics
from utils.dataset_utils import PromptTrainDataset, DenoiseTestDataset, DerainDehazeDataset, DehazeDataset, LolDataset, GoproDataset
import os

class MMAiOModel(LightningModule):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.net = PromptIR()
        self.loss_fn = nn.L1Loss()
        self.psnr = torchmetrics.image.PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = torchmetrics.image.StructuralSimilarityIndexMeasure(data_range=1.0)
        self.val_psnr_values = {0: [], 1: [], 2: [], 3:[], 4:[]}
        self.val_ssim_values = {0: [], 1: [], 2: [], 3:[], 4:[]}
        self.val_task_names = ['denoise', 'derain', 'dehaze', 'gopro', 'lol']


    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        ([clean_name, de_id], degrad_patch, clean_patch) = batch
        restored = self.net(degrad_patch)
        loss = self.loss_fn(restored, clean_patch)
        psnr_value = self.psnr(restored, clean_patch)
        ssim_value = self.ssim(restored, clean_patch)
        lr = self.optimizers().param_groups[0]['lr']
        self.log("lr", lr, on_step=True, prog_bar=True, logger=True)
        self.log("train_psnr", psnr_value, prog_bar=True, logger=True)
        self.log("train_ssim", ssim_value, prog_bar=True, logger=True)
        self.log("train_loss", loss, on_step=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        ([clean_name], degrad_patch, clean_patch) = batch
        restored = self.net(degrad_patch)
        psnr_value = self.psnr(restored, clean_patch)
        ssim_value = self.ssim(restored, clean_patch)
        
        task_name = self.val_task_names[dataloader_idx]
        self.val_psnr_values[dataloader_idx].append(psnr_value)
        self.val_ssim_values[dataloader_idx].append(ssim_value)
        
        self.log(f"val_psnr_{task_name}", psnr_value, on_step=True, prog_bar=True, logger=True, sync_dist=True)
        self.log(f"val_ssim_{task_name}", ssim_value, on_step=True, prog_bar=True, logger=True, sync_dist=True)
        
        return psnr_value, ssim_value
    
    def on_validation_epoch_end(self):
        avg_psnr_0 = torch.stack(self.val_psnr_values[0]).mean() if self.val_psnr_values[0] else torch.tensor(0.0)
        avg_psnr_1 = torch.stack(self.val_psnr_values[1]).mean() if self.val_psnr_values[1] else torch.tensor(0.0)
        avg_psnr_2 = torch.stack(self.val_psnr_values[2]).mean() if self.val_psnr_values[2] else torch.tensor(0.0)
        avg_psnr_3 = torch.stack(self.val_psnr_values[3]).mean() if self.val_psnr_values[3] else torch.tensor(0.0)
        avg_psnr_4 = torch.stack(self.val_psnr_values[4]).mean() if self.val_psnr_values[4] else torch.tensor(0.0)
        avg_ssim_0 = torch.stack(self.val_ssim_values[0]).mean() if self.val_ssim_values[0] else torch.tensor(0.0)
        avg_ssim_1 = torch.stack(self.val_ssim_values[1]).mean() if self.val_ssim_values[1] else torch.tensor(0.0)
        avg_ssim_2 = torch.stack(self.val_ssim_values[2]).mean() if self.val_ssim_values[2] else torch.tensor(0.0)
        avg_ssim_3 = torch.stack(self.val_ssim_values[3]).mean() if self.val_ssim_values[3] else torch.tensor(0.0)
        avg_ssim_4 = torch.stack(self.val_ssim_values[4]).mean() if self.val_ssim_values[4] else torch.tensor(0.0)
        
        avg_val_psnr = (avg_psnr_0 + avg_psnr_1 + avg_psnr_2 + avg_psnr_3 + avg_psnr_4) / 5
        self.log('avg_val_psnr', avg_val_psnr, prog_bar=True, logger=True, sync_dist=True)
        avg_val_ssim = (avg_ssim_0 + avg_ssim_1 + avg_ssim_2 + avg_ssim_3 + avg_ssim_4) / 5
        self.log('avg_val_ssim', avg_val_ssim, prog_bar=True, logger=True, sync_dist=True)
        
        for key in self.val_psnr_values:
            self.val_psnr_values[key].clear()
        for key in self.val_ssim_values:
            self.val_ssim_values[key].clear()

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=2e-4)
        scheduler = schedulers.LinearWarmupCosineAnnealingLR(optimizer=optimizer, warmup_epochs=15, max_epochs=180)
        return [optimizer], [scheduler]

def main():
    torch.set_float32_matmul_precision('high')
    logger = TensorBoardLogger(save_dir="logs", name=opt.logs_name, version=0)
    trainset = PromptTrainDataset(opt)
    trainloader = DataLoader(trainset, batch_size=opt.batch_size, pin_memory=True, shuffle=True,  drop_last=True, num_workers=opt.num_workers)

    denoise_splits = ["bsd68/"]
    derain_splits = ["Rain100L/"]
    denoise_tests = []
    derain_tests = []
    base_path = opt.denoise_path
    opt.denoise_path = os.path.join(base_path,denoise_splits[0])
    denoise_testset = DenoiseTestDataset(opt)
    valset_denoise = denoise_testset
    
    derain_base_path = opt.derain_path
    opt.derain_path = os.path.join(derain_base_path,derain_splits[0])
    derain_set = DerainDehazeDataset(opt)
    valset_derain = derain_set
    dehaze_set = DehazeDataset(opt)
    valset_dehaze = dehaze_set
    gopro_set = GoproDataset(opt)
    valset_gopro = gopro_set
    lol_set = LolDataset(opt)
    valset_lol = lol_set  

    valloader_denoise = DataLoader(valset_denoise, batch_size=1, pin_memory=True,shuffle=False, num_workers=opt.num_workers)
    valloader_derain = DataLoader(valset_derain, batch_size=1, pin_memory=True,shuffle=False, num_workers=opt.num_workers)
    valloader_dehaze = DataLoader(valset_dehaze, batch_size=1, pin_memory=True,shuffle=False, num_workers=opt.num_workers)
    valloader_gopro = DataLoader(valset_gopro, batch_size=1, pin_memory=True,shuffle=False, num_workers=opt.num_workers)
    valloader_lol = DataLoader(valset_lol, batch_size=1, pin_memory=True,shuffle=False, num_workers=opt.num_workers)

    checkpoint_callback = ModelCheckpoint(dirpath=opt.ckpt_dir, every_n_epochs=1, save_top_k=-1)
    model = MMAiOModel(opt)
    trainer = Trainer(max_epochs=opt.every_epochs, accelerator="gpu", devices=[0,1], strategy="ddp_find_unused_parameters_true", logger=logger, callbacks=[checkpoint_callback], enable_progress_bar=True)

    trainer.fit(model=model, train_dataloaders=trainloader, val_dataloaders=[valloader_denoise, valloader_derain, valloader_dehaze, valloader_gopro, valloader_lol],ckpt_path=opt.input_ckpt)

if __name__ == '__main__':
    main()
