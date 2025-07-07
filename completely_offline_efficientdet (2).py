import argparse
import os
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import CocoDataset, Resizer, Normalizer, Augmenter, collater
from model import EfficientDet
from torch.autograd import Variable
import shutil
import numpy as np
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau

def get_args():
    parser = argparse.ArgumentParser('EfficientDet Training Script')
    parser.add_argument('-c', '--config', type=str, default='config.yml')
    parser.add_argument('-p', '--project', type=str, default='coco')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--verbose', type=int, default=1)
    args = parser.parse_args()
    return args

def save_checkpoint(state, is_best, filename='checkpoint.pth'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth')

def train(opt):
    with open(opt.config) as f:
        config = yaml.safe_load(f)

    train_params = config['train_params']
    project_name = opt.project
    batch_size = opt.batch_size
    num_epochs = opt.num_epochs
    num_workers = opt.num_workers
    lr = opt.lr
    weight_decay = opt.weight_decay
    verbose = opt.verbose

    # Setup device (CPU only)
    device = torch.device('cpu')

    # Setup dataset
    dataset_train = CocoDataset(
        root_dir=config['dataset']['train_root_dir'],
        set_name=config['dataset']['train_set_name'],
        transform=Augmenter(),
        visual_debug=False
    )
    dataset_val = CocoDataset(
        root_dir=config['dataset']['val_root_dir'],
        set_name=config['dataset']['val_set_name'],
        transform=Normalizer(),
        visual_debug=False
    )

    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, batch_size, drop_last=True
    )

    dataloader_train = DataLoader(
        dataset_train,
        batch_sampler=batch_sampler_train,
        num_workers=num_workers,
        collate_fn=collater
    )

    dataloader_val = DataLoader(
        dataset_val,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collater
    )

    # Setup model
    model = EfficientDet(num_classes=dataset_train.num_classes, 
                         compound_coef=config['compound_coef'])
    model = model.to(device)

    # Setup optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, patience=3, verbose=True)

    # Resume from checkpoint
    if opt.resume is not None:
        checkpoint = torch.load(opt.resume, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        best_loss = checkpoint['best_loss']
    else:
        start_epoch = 0
        best_loss = float('inf')

    # Setup tensorboard
    writer = SummaryWriter(f'logs/{project_name}')

    # Training loop
    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_loss = []
        
        progress_bar = tqdm(dataloader_train)
        for iter_num, data in enumerate(progress_bar):
            optimizer.zero_grad()

            imgs = data['img'].to(device)
            annot = data['annot'].to(device)

            classification_loss, regression_loss = model(imgs, annot)
            classification_loss = classification_loss.mean()
            regression_loss = regression_loss.mean()

            loss = classification_loss + regression_loss
            if loss == 0 or not torch.isfinite(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()

            epoch_loss.append(float(loss))

            progress_bar.set_description(
                'Epoch: {}/{}. Iteration: {}/{}. Cls loss: {:.5f}. Reg loss: {:.5f}. Total loss: {:.5f}'.format(
                    epoch + 1, num_epochs, iter_num + 1, len(dataloader_train),
                    classification_loss.item(), regression_loss.item(), loss.item()
                )
            )

            # Tensorboard logging
            writer.add_scalar('Train/Classification_loss', classification_loss.item(), 
                            epoch * len(dataloader_train) + iter_num)
            writer.add_scalar('Train/Regression_loss', regression_loss.item(), 
                            epoch * len(dataloader_train) + iter_num)

        # Validation
        if verbose > 0:
            model.eval()
            val_loss = []
            val_iter = tqdm(dataloader_val, desc='Validating')
            for iter_num, data in enumerate(val_iter):
                with torch.no_grad():
                    imgs = data['img'].to(device)
                    annot = data['annot'].to(device)

                    classification_loss, regression_loss = model(imgs, annot)
                    classification_loss = classification_loss.mean()
                    regression_loss = regression_loss.mean()

                    loss = classification_loss + regression_loss
                    if loss == 0 or not torch.isfinite(loss):
                        continue

                    val_loss.append(float(loss))
                    val_iter.set_description(
                        'Val. Cls loss: {:.5f}. Reg loss: {:.5f}. Total loss: {:.5f}'.format(
                            classification_loss.item(), regression_loss.item(), loss.item()
                        )
                    )

            val_loss = np.mean(val_loss)
            scheduler.step(val_loss)

            # Save checkpoint
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_loss': best_loss,
                'optimizer': optimizer.state_dict(),
            }, val_loss < best_loss)

            if val_loss < best_loss:
                best_loss = val_loss

            writer.add_scalar('Val/Loss', val_loss, epoch)

    writer.close()

if __name__ == '__main__':
    opt = get_args()
    train(opt)
