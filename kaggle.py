import textwrap

with open("train.py", "w") as f:
    f.write(textwrap.dedent("""
        import subprocess
        from tqdm import tqdm
        import sys
        import os
        import torch
        import torch.nn as nn
        import torch.optim as optim
        import numpy as np
        from torch.utils.data import DataLoader
        from torchvision import transforms
        import lightning.pytorch as pl
        from lightning.pytorch.callbacks import ModelCheckpoint
        import argparse
        from model import PromptIR
        from utils import LinearWarmupCosineAnnealingLR
        from preprocess import save_image_tensor, PromptTrainDataset

        class PromptIRModel(pl.LightningModule):
            def __init__(self):
                super().__init__()
                self.net = PromptIR(decoder=True)
                self.loss_fn = nn.L1Loss()
                self.epoch_loss_sum = 0.0
                self.epoch_batch_count = 0

            def forward(self, x):
                return self.net(x)

            def training_step(self, batch, batch_idx):
                [clean_name, de_id], degrad_patch, clean_patch = batch
                if self.epoch_batch_count <= 20:
                    save_image_tensor(clean_patch, f'temp/{self.epoch_batch_count}_before.png')
                restored = self.net(degrad_patch)
                loss = self.loss_fn(restored, clean_patch)
                if self.epoch_batch_count <= 20:
                    save_image_tensor(restored, f'temp/{self.epoch_batch_count}_after.png')
                self.epoch_loss_sum += loss.item()
                self.epoch_batch_count += 1
                return loss

            def lr_scheduler_step(self, scheduler, metric):
                scheduler.step(self.current_epoch)

            def configure_optimizers(self):
                optimizer = optim.AdamW(self.parameters(), lr=5e-5)
                scheduler = LinearWarmupCosineAnnealingLR(
                    optimizer=optimizer,
                    warmup_epochs=0,
                    max_epochs=40
                )
                return [optimizer], [scheduler]

            def on_train_epoch_start(self):
                self.epoch_loss_sum = 0.0
                self.epoch_batch_count = 0
                os.makedirs("temp", exist_ok=True)

            def on_train_epoch_end(self):
                avg = self.epoch_loss_sum / max(1, self.epoch_batch_count)
                epoch = self.current_epoch + 1
                with open("loss.txt", "a") as f:
                    f.write(f"Epoch {epoch:03d}: {avg:.6f}\\n")

        def main():
            parser = argparse.ArgumentParser()
            parser.add_argument('--cuda', type=int, default=0)
            parser.add_argument('--epochs', type=int, default=40)
            parser.add_argument('--batch_size', type=int, default=1)
            parser.add_argument('--lr', type=float, default=2e-4)
            parser.add_argument('--de_type', nargs='+', default=[
                'denoise_15', 'denoise_25', 'denoise_50', 'derain', 'dehaze'
            ])
            parser.add_argument('--patch_size', type=int, default=128)
            parser.add_argument('--num_workers', type=int, default=16)
            parser.add_argument('--root_dir', type=str, default='/kaggle/input/hw4-dataset/hw4_realse_dataset')
            parser.add_argument('--clean_dir', type=str, default='/kaggle/input/hw4-dataset/hw4_realse_dataset/train/clean/')
            parser.add_argument('--degraded_dir', type=str, default='/kaggle/input/hw4-dataset/hw4_realse_dataset/train/degraded/')
            parser.add_argument('--data_file_dir', type=str, default='/kaggle/input/hw4-dataset/hw4_realse_dataset/')
            parser.add_argument('--derain_dir', type=str, default='/kaggle/input/hw4-dataset/hw4_realse_dataset/train/')
            parser.add_argument('--output_path', type=str, default="output/")
            parser.add_argument('--ckpt_dir', type=str, default="ckpt/")
            parser.add_argument('--num_gpus', type=int, default=1)
            parser.add_argument('--wblogger', type=str, default="promptir")

            args = parser.parse_args()
            print("device is", 'cuda' if torch.cuda.is_available() else 'cpu')

            transform = transforms.Compose([transforms.ToTensor()])
            trainset = PromptTrainDataset(args)
            trainloader = DataLoader(
                trainset,
                batch_size=args.batch_size,
                pin_memory=True,
                shuffle=True,
                drop_last=True
            )

            model = PromptIRModel().cuda()                            
            ckpt_path = '/kaggle/working/ckpt/epoch=45-step=147200.ckpt'
            model = PromptIRModel.load_from_checkpoint(ckpt_path).cuda()

            checkpoint_callback = ModelCheckpoint(
                dirpath=args.ckpt_dir,
                every_n_epochs=1,
                save_top_k=-1
            )
            trainer = pl.Trainer(
                max_epochs=args.epochs,
                accelerator="gpu",
                devices=args.num_gpus,
                callbacks=[checkpoint_callback]
            )
            trainer.fit(
                model=model,
                train_dataloaders=trainloader,
                ckpt_path=ckpt_path
            )

        if __name__ == '__main__':
            main()
    """))
