# Author: Akira Kudo
# Created: 2024/10/29
# Last Updated: 2024/11/20

import os
import re

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from wgan.model.wgan_gp import Discriminator, Generator
from wgan.trainer.trainer import Trainer

def find_latest_checkpoint(chkpt_dir : str, 
                           stop_if_not_found : bool=True):
    """
    Within the subfolders 'netD' and 'netG' under chkpt_dir,
    find the latest checkpoint custom file objects, returning
    their paths.

    :param str chkpt_dir: Directory holding 'netD' and 'netG'
    directories which hold the checkpoint files.
    :param bool stop_if_not_found: Stop if no checkpoint is found.
    Defaults to true.

    :returns str netD_chkpt_path: Path to the found discriminator 
    checkpoint file.
    :returns str netG_chkpt_path: Path to the found generator 
    checkpoint file.
    """
    NET_D_DIRNAME, NET_G_DIRNAME = "netD", "netG"
    latest_D_chkpt, latest_G_chkpt = None, None
    
    if os.path.isdir(chkpt_dir):
        netD_dir = os.path.join(chkpt_dir, NET_D_DIRNAME)
        netG_dir = os.path.join(chkpt_dir, NET_G_DIRNAME)

        if os.path.isdir(netD_dir) and os.path.isdir(netG_dir):
            allfiles = list(os.listdir(netD_dir))
            matches = [re.search(r'\d+', file) 
                       for file in os.listdir(netD_dir)]
            stepnums = [int(m.group()) for m in matches]
            latest_D_stepnum = max(stepnums)
            # get the latest checkpoint path for both componentss
            latest_D_chkpt = os.path.join(
                netD_dir, allfiles[stepnums.index(latest_D_stepnum)]
                )
            latest_G_chkpt = os.path.join(
                netG_dir, os.path.basename(latest_D_chkpt).replace('D', 'G')
                )
    if stop_if_not_found and latest_D_chkpt is None or latest_G_chkpt is None:
        raise Exception("Checkpoints not found; terminating.")
    else:
        print(f"Found latest checkpoints: \n{latest_D_chkpt}\n{latest_G_chkpt}")

    return latest_D_chkpt, latest_G_chkpt


if __name__ == "__main__":
    initial_dir = os.getcwd()
    os.chdir(os.path.dirname(__file__))
    
    # latest_D_chkpt, latest_G_chkpt = find_latest_checkpoint("../data/model/checkpoints")
    
    try:
        # Loading data
        data = np.load("../data/training/normalized-training-open-64ch.npy")
        data = torch.tensor(data)

        # Data handling objects
        if torch.cuda.is_available():
            print("hello GPU")
        else:
            print("sadge")
        
        device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
        dataloader = DataLoader(
            TensorDataset(data),
            batch_size=32,
            shuffle=True
        )

        def weights_init(model):
            for m in model.modules():
                if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d, nn.BatchNorm1d)):
                    nn.init.normal_(m.weight.data, 0.0, 0.02)


        netD = Discriminator()
        netG = Generator()

        weights_init(netD)
        weights_init(netG)

        if torch.cuda.device_count() > 1:
            print("Using", torch.cuda.device_count(), "GPUs")
            netD = nn.DataParallel(netD)
            netG = nn.DataParallel(netG)

        netD.to("cuda" if torch.cuda.is_available() else "cpu")
        netG.to("cuda" if torch.cuda.is_available() else "cpu")

        optD = optim.Adam(netD.parameters(), 0.0001, (0.5, 0.99))
        optG = optim.Adam(netG.parameters(), 0.0001, (0.5, 0.99))


        # Start training
        trainer = Trainer(
            # netD=netD.module, # ORIGINALLY
            netD=netD,
            # netG=netG.module, # ORIGINALLY
            netG=netG,
            optD=optD,
            optG=optG,
            n_dis=5,
            num_steps=1000000,
            lr_decay='linear',
            dataloader=dataloader,
            save_steps=5000,
            print_steps=100,
            log_dir='../data/model',
            device=device
            )
        trainer.train()
    
    except Exception as e:
        print(e)
    finally:
        os.chdir(initial_dir)
    


    # # epoch = global step / 2




    # # !pip install gdown
    # # !gdown --id 1nZeG8_lE6CQk_kcs3Uicrt-OFN9XQMdp
    # # ! unzip /kaggle/working/normalized-training-open-64ch.zip

    # # !gdown --id 1iAo5hpi7zXvoMOOPNpd6igaypQsQCwwd
    # # ! unzip /kaggle/working/open.zip

    # # data = np.load("/kaggle/working/normalized-training-open-64ch.npy")
    # # data = torch.tensor(data).detach()

    # # gen = netG.module.generate_signals(42)

    # # np.save('generated-open-3', gen.cpu().detach().numpy())