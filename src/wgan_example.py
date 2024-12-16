# Author: Akira Kudo
# Created: 2024/11/22
# Last Updated: 2024/12/16

""" A file showing an example generating signals using a trained WGAN Generator."""

import numpy as np

from wgan.model.wgan_gp import Generator

if __name__ == "__main__":
    # To generate signals:
    # - instantiate a Generator class object
    # - load the trained generator weights
    # - call its generate_signals(num_signals, device=None)
    MODEL_PATH = r"C:\Users\mashi\Desktop\VSCode\python\MINT\eeg-wgan\models\trained_akira\netG_23500_steps.pth"
    NUM_SIGNALS = 5
    SAVE_SIGNALS = True

    netG = Generator() 
    netG.restore_checkpoint(ckpt_file=MODEL_PATH)
    fake_signals = netG.generate_signals(num_signals=NUM_SIGNALS).detach().numpy()

    print("Generated {} samples; overall shape of ({}, {}).".format(
        NUM_SIGNALS, *fake_signals.shape))
    
    if SAVE_SIGNALS:
        print("Saved the fake signals...", end="")
        np.save("./fake_signal1.npy", fake_signals)
        print("DONE!")