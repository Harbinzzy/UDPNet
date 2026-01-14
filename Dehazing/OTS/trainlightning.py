import os
import torch
from data import train_dataloader
from data import valid_dataloader
import torch.nn.functional as F
import torch.nn as nn
import torchmetrics
from torch.utils.tensorboard import SummaryWriter
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from warmup_scheduler import GradualWarmupScheduler
from pytorch_lightning.callbacks import Callback
import torch.nn.functional as f

torch.set_float32_matmul_precision('high')

class PeriodicCheckpointCallback(Callback):
    def __init__(self, save_dir, interval=50):
        self.save_dir = save_dir
        self.interval = interval

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        if epoch % self.interval == 0:
            ckpt_path = os.path.join(self.save_dir, f"epoch_{epoch:02d}.ckpt")
            trainer.save_checkpoint(ckpt_path)

class HazeRemovalModule(pl.LightningModule):
    def __init__(self, model, args):
        super().__init__()
        self.model = model
        self.args = args
        self.criterion = torch.nn.L1Loss()
        self.psnr = torchmetrics.image.PeakSignalNoiseRatio(data_range=1.0)
        self.save_hyperparameters(ignore=['model'])
        self.version = 0
        
    def forward(self, x):
        return self.model(x)
        
    def fft_loss(self, pred, label):
        pred_fft = torch.fft.fft2(pred, dim=(-2, -1))
        label_fft = torch.fft.fft2(label, dim=(-2, -1))
        pred_fft = torch.stack((pred_fft.real, pred_fft.imag), -1)
        label_fft = torch.stack((label_fft.real, label_fft.imag), -1)
        return self.criterion(pred_fft, label_fft)
    
    def training_step(self, batch, batch_idx):
        input_img, label_img = batch
        pred_img = self(input_img)
        
        label_img2 = F.interpolate(label_img, scale_factor=0.5, mode='bilinear')
        label_img4 = F.interpolate(label_img, scale_factor=0.25, mode='bilinear')
        l1 = self.criterion(pred_img[0], label_img4)
        l2 = self.criterion(pred_img[1], label_img2)
        l3 = self.criterion(pred_img[2], label_img)
        loss_content = l1 + l2 + l3
        
        f1 = self.fft_loss(pred_img[0], label_img4)
        f2 = self.fft_loss(pred_img[1], label_img2)
        f3 = self.fft_loss(pred_img[2], label_img)
        loss_fft = f1 + f2 + f3
        
        loss = loss_content + 0.1 * loss_fft
        
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']

        psnr_value = self.psnr(pred_img[2], label_img)
        self.log("train_psnr", psnr_value, prog_bar=True, logger=True)
        self.log('learning_rate', current_lr, prog_bar=True)
        self.log('train_loss_content', loss_content, prog_bar=True)
        self.log('train_loss_fft', loss_fft, prog_bar=True)
        self.log('train_loss', loss, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        input_img, label_img = batch
        factor = 8
        
        h, w = input_img.shape[2], input_img.shape[3]
        H, W = ((h+factor)//factor)*factor, ((w+factor)//factor*factor)
        padh = H-h if h%factor!=0 else 0
        padw = W-w if w%factor!=0 else 0
        input_img = f.pad(input_img, (0, padw, 0, padh), 'reflect')
        
        pred = self(input_img)[2]
        pred = pred[:,:,:h,:w]
        
        pred_clip = torch.clamp(pred, 0, 1)
        
        psnr = self.psnr(pred_clip, label_img)
        
        self.log('val_psnr', psnr, prog_bar=True, sync_dist=True)
        return psnr
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        warmup_epochs = 1
        scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.args.num_epoch - warmup_epochs,
            eta_min=1e-6
        )
        
        if self.args.resume and "epoch" or "last" in self.args.resume:
            scheduler = scheduler_cosine
        else:
            scheduler = GradualWarmupScheduler(
                optimizer,
                multiplier=1,
                total_epoch=warmup_epochs,
                after_scheduler=scheduler_cosine
            )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch"
            }
        }

def _train(model, args):
    lightning_module = HazeRemovalModule(model, args)
    
    train_loader = train_dataloader(args.data_dir, args.batch_size, args.num_worker)
    val_loader = valid_dataloader(args.data_dir, batch_size=1, num_workers=0)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.model_save_dir,
        filename='epoch_{epoch:02d}_psnr_{val_psnr:.2f}',
        monitor='val_psnr',
        mode='max',
        save_top_k=3,
        save_last=True
    )
    periodic_ckpt_callback = PeriodicCheckpointCallback(
        save_dir=args.model_save_dir,
        interval=args.save_freq
    )
    
    logger = TensorBoardLogger(
        save_dir='logs',
        default_hp_metric=False,
        name=args.logs
    )
    
    trainer = pl.Trainer(
        max_epochs=args.num_epoch,
        log_every_n_steps=1,
        accelerator='gpu',
        devices=-1,
        strategy="ddp_find_unused_parameters_true",
        callbacks=[checkpoint_callback, periodic_ckpt_callback],
        logger=logger,
        gradient_clip_val=0.01,
        check_val_every_n_epoch=args.valid_freq
    )
    
    if args.resume:
        trainer.fit(
            lightning_module,
            train_loader,
            val_loader,
            ckpt_path = args.resume
        )
    else:
        trainer.fit(lightning_module, train_loader, val_loader)
    
    trainer.save_checkpoint(os.path.join(args.model_save_dir, 'Final.ckpt'))
    print(f"训练完成！最终模型保存在 {args.model_save_dir}/Final.ckpt")
