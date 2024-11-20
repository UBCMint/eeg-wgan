# Author: Joshua Park
# Modified by: Akira Kudo
# Created: 2024/11/04
# Last Updated: 2024/11/18

import matplotlib.pyplot as plt
import numpy as np
import torch

def compute_psd(data, fs, nperseg=256, noverlap=None):
    """
    Compute Power Spectral Density (PSD) using the Welch method.

    Parameters:
        data (array): EEG data array with shape (n_channels, n_samples).
        fs (float): Sampling frequency.
        nperseg (int): Length of each segment for PSD estimation.
        noverlap (int): Number of overlapping samples between segments.

    Returns:
        freqs (array): Frequency values.
        psd (array): Power Spectral Density values.
    """
    n_channels, n_samples = data.shape
    psd = np.zeros((n_channels, nperseg // 2 + 1))

    for ch_idx in range(n_channels):
        f, Pxx = plt.psd(data[ch_idx].cpu(), Fs=fs, NFFT=256, noverlap=128, window=np.hanning(256), scale_by_freq=True)
        # Add a small epsilon to avoid zero values
        psd[ch_idx] = Pxx + 1e-10

    return f, psd


def average_across_arrays(generated_data):
    return generated_data.mean(dim=0)

def plot_everything(generated_data, gen_err, critic_err):
    generated_data = generated_data.detach()

    # plotting generated data
    values = generated_data[0, 0, :]
    plt.plot(values.tolist())
    plt.show()

    # plotting PSD
    psd = get_fft_feature_train(generated_data.cpu())
    plt.plot(psd[0,0])
    plt.show()
#     averaged_data = average_across_arrays(generated_data)
#     freqs, psd = compute_psd(averaged_data, 160.0)
#     plt.figure(figsize=(10, 6))  # Add this line to create a single figure
#     for ch_idx in range(1):
#         plt.semilogy(freqs, psd[ch_idx], label=f'Channel {ch_idx + 1}')
#     plt.xlabel('Frequency (Hz)')
#     plt.ylabel('Power/Frequency (dB/Hz)')
#     plt.show()

    # plotting G vs D losses
    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(gen_err,label="Generator")
    plt.plot(critic_err,label="Critic")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


def get_fft_feature_train(data, nperseg=256, noverlap=128, channels=64):
    all_fft = []
    device = data.device

    # Create window function
    window = torch.hann_window(nperseg, dtype=torch.float, device=device)

    for x in data:
        avg_psds_db = []
        
        for ch in range(channels):
            chx = x[ch]

            # Separate x into overlapping segments
            x_segs = chx.unfold(0, nperseg, nperseg - noverlap)

            # Apply window function to each segment
            windowed_segs = x_segs * window

            # Compute power spectral density for each windowed segment
            seg_psds = torch.fft.rfft(windowed_segs, dim=1)
            seg_psds = torch.abs(seg_psds)**2

            # Average PSDs over all segments
            avg_psds = torch.mean(seg_psds, axis=0)

            # Convert to decibels
            avg_psds_db.append(torch.log10(avg_psds + 1e-10))

        avg_psds_db = torch.stack(avg_psds_db)
        all_fft.append(avg_psds_db)

    all_fft = torch.stack(all_fft, dim=0).to(device)
    return all_fft