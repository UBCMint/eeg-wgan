# Author: Joshua Park
# Modified by: Akira Kudo
# Created: 2024/11/01
# Last Updated: 2024/11/20

"""
Base class implementation of WGAN-GP.
"""

import torch
from torch import autograd
import torch.nn as nn

from wgan.model.base_model import BaseModel
from wgan.model.loss import hinge_loss_dis, hinge_loss_gen, minimax_loss_dis, minimax_loss_gen, ns_loss_gen, wasserstein_loss_dis, wasserstein_loss_gen
from wgan.model.residual_block import DBlock, GBlock
from wgan.utils.visualize_psd import plot_everything

"""
Implementation of Base GAN models.
"""

class BaseGenerator(BaseModel):
    r"""
    Base class for a generic unconditional generator model.

    Attributes:
        nz (int): Noise dimension for upsampling.
        ngf (int): Variable controlling generator feature map sizes.
        sequence_length (int): Starting width for upsampling generator output to a signal.
        loss_type (str): Name of loss to use for GAN loss.
    """
    def __init__(self, nz, ngf, sequence_length, loss_type, **kwargs):
        super().__init__(**kwargs)
        self.nz = nz
        self.ngf = ngf
        self.sequence_length = sequence_length
        self.loss_type = loss_type

    def generate_signals(self, num_signals, device=None):
        r"""
        Generates num_signals randomly.

        Args:
            num_signals (int): Number of signals to generate
            device (torch.device): Device to send signals to.

        Returns:
            Tensor: A batch of generated signals.
        """
        if device is None:
            device = self.device

        noise = torch.randn((num_signals, self.nz), device=device)
        fake_signals = self.forward(noise)

        return fake_signals

    def compute_gan_loss(self, output):
        r"""
        Computes GAN loss for generator.

        Args:
            output (Tensor): A batch of output logits from the discriminator of shape (N, 1).

        Returns:
            Tensor: A batch of GAN losses for the generator.
        """
        # Compute loss and backprop
        if self.loss_type == "gan":
            errG = minimax_loss_gen(output)

        elif self.loss_type == "ns":
            errG = ns_loss_gen(output)

        elif self.loss_type == "hinge":
            errG = hinge_loss_gen(output)

        elif self.loss_type == "wasserstein":
            errG = wasserstein_loss_gen(output)

        else:
            raise ValueError("Invalid loss_type {} selected.".format(self.loss_type))

        return errG

    def train_step(self,
                   real_batch,
                   netD,
                   optG,
                   log_data,
                   device=None,
                   global_step=None,
                   **kwargs):
        r"""
        Takes one training step for G.

        Args:
            real_batch (Tensor): A batch of real signals of shape (N, C, H, W). Used for obtaining current batch size.
            netD (nn.Module): Discriminator model for obtaining losses.
            optG (Optimizer): Optimizer for updating generator's parameters.
            log_data (dict): A dict mapping name to values for logging uses.
            device (torch.device): Device to use for running the model.
            global_step (int): Variable to sync training, logging and checkpointing. Useful for dynamic changes to model amidst training.

        Returns:
            Returns MetricLog object containing updated logging variables after 1 training step.

        """
        self.zero_grad()

        # Get only batch size from real batch
        batch_size = real_batch[0].shape[0]

        # Produce fake signals
        fake_signals = self.generate_signals(num_signals=batch_size, device=device)

        # Compute output logit of D thinking signal real
        output = netD(fake_signals)

        # Compute loss
        errG = self.compute_gan_loss(output=output)

        # Backprop and update gradients
        errG.backward()
        optG.step()

        # Log statistics
        log_data.add_metric('errG', errG, group='loss')

        return log_data


class BaseDiscriminator(BaseModel):
    r"""
    Base class for a generic unconditional discriminator model.

    Attributes:
        ndf (int): Variable controlling discriminator feature map sizes.
        loss_type (str): Name of loss to use for GAN loss.
    """
    def __init__(self, ndf, loss_type, **kwargs):
        super().__init__(**kwargs)
        self.ndf = ndf
        self.loss_type = loss_type

    def compute_gan_loss(self, output_real, output_fake):
        r"""
        Computes GAN loss for discriminator.

        Args:
            output_real (Tensor): A batch of output logits of shape (N, 1) from real signals.
            output_fake (Tensor): A batch of output logits of shape (N, 1) from fake signals.

        Returns:
            errD (Tensor): A batch of GAN losses for the discriminator.
        """
        # Compute loss for D
        if self.loss_type == "gan" or self.loss_type == "ns":
            errD = minimax_loss_dis(output_fake=output_fake, output_real=output_real)

        elif self.loss_type == "hinge":
            errD = hinge_loss_dis(output_fake=output_fake, output_real=output_real)

        elif self.loss_type == "wasserstein":
            errD = wasserstein_loss_dis(output_fake=output_fake, output_real=output_real)

        else:
            raise ValueError("Invalid loss_type selected.")

        return errD

    def compute_probs(self, output_real, output_fake):
        r"""
        Computes probabilities from real/fake signals logits.

        Args:
            output_real (Tensor): A batch of output logits of shape (N, 1) from real signals.
            output_fake (Tensor): A batch of output logits of shape (N, 1) from fake signals.

        Returns:
            tuple: Average probabilities of real/fake signal considered as real for the batch.
        """
        D_x = torch.sigmoid(output_real).mean().item()
        D_Gz = torch.sigmoid(output_fake).mean().item()

        return D_x, D_Gz

    def train_step(self,
                   real_batch,
                   netG,
                   optD,
                   log_data,
                   device=None,
                   global_step=None,
                   **kwargs):
        r"""
        Takes one training step for D.

        Args:
            real_batch (Tensor): A batch of real signals of shape (N, C, H, W).
            loss_type (str): Name of loss to use for GAN loss.
            netG (nn.Module): Generator model for obtaining fake signals.
            optD (Optimizer): Optimizer for updating discriminator's parameters.
            device (torch.device): Device to use for running the model.
            log_data (dict): A dict mapping name to values for logging uses.
            global_step (int): Variable to sync training, logging and checkpointing.
                Useful for dynamic changes to model amidst training.

        Returns:
            MetricLog: Returns MetricLog object containing updated logging variables after 1 training step.
        """
        self.zero_grad()
        real_signals, real_labels = real_batch
        batch_size = real_signals.shape[0]  # Match batch sizes for last iter

        # Produce logits for real signals
        output_real = self.forward(real_signals)

        # Produce fake signals
        fake_signals = netG.generate_signals(num_signals=batch_size, device=device).detach()

        # Produce logits for fake signals
        output_fake = self.forward(fake_signals)

        # Compute loss for D
        errD = self.compute_gan_loss(output_real=output_real, output_fake=output_fake)

        # Backprop and update gradients
        errD.backward()
        optD.step()

        # Compute probabilities
        D_x, D_Gz = self.compute_probs(output_real=output_real, output_fake=output_fake)

        # Log statistics for D once out of loop
        log_data.add_metric('errD', errD.item(), group='loss')
        log_data.add_metric('D(x)', D_x, group='prob')
        log_data.add_metric('D(G(z))', D_Gz, group='prob')

        return log_data


class WGANGPBaseGenerator(BaseGenerator):
    r"""
    ResNet backbone generator for WGAN-GP.

    Attributes:
        nz (int): Noise dimension for upsampling.
        ngf (int): Variable controlling generator feature map sizes.
        sequence_length (int): Starting width for upsampling generator output to an signal.
        loss_type (str): Name of loss to use for GAN loss.
    """
    def __init__(self,
                 nz,
                 ngf,
                 sequence_length,
                 loss_type='wasserstein',
                 **kwargs):
        super().__init__(nz=nz,
                         ngf=ngf,
                         sequence_length=sequence_length,
                         loss_type=loss_type,
                         **kwargs)

    def train_step(self,
                   real_batch,
                   netD,
                   optG,
                   log_data,
                   device=None,
                   global_step=None,
                   **kwargs):
        r"""
        Takes one training step for G.

        Args:
            real_batch (Tensor): A batch of real signals of shape (N, C, L).
                Used for obtaining current batch size.
            netD (nn.Module): Discriminator model for obtaining losses.
            optG (Optimizer): Optimizer for updating generator's parameters.
            log_data (MetricLog): An object to add custom metrics for visualisations.
            device (torch.device): Device to use for running the model.
            global_step (int): Variable to sync training, logging and checkpointing.
                Useful for dynamic changes to model amidst training.

        Returns:
            MetricLog: Returns MetricLog object containing updated logging variables after 1 training step.

        """
        self.zero_grad()

        # Get only batch size from real batch
        batch_size = real_batch[0].shape[0]

        # Produce fake signals
        fake_signals = self.generate_signals(num_signals=batch_size, device=device)

        # Compute output logit of D thinking signal real
        output = netD(fake_signals)

        # Compute loss
        errG = self.compute_gan_loss(output)

        # Backprop and update gradients
        errG.backward()
        optG.step()

        # Log statistics
        log_data.add_metric('errG', errG, group='loss')

        return log_data


class WGANGPBaseDiscriminator(BaseDiscriminator):
    r"""
    ResNet backbone discriminator for WGAN-GP.

    Attributes:
        ndf (int): Variable controlling discriminator feature map sizes.
        loss_type (str): Name of loss to use for GAN loss.
        gp_scale (float): Lamda parameter for gradient penalty.
    """
    def __init__(self, ndf, loss_type='wasserstein', gp_scale=10.0, **kwargs):
        super().__init__(ndf=ndf, loss_type=loss_type, **kwargs)
        self.gp_scale = gp_scale

    def train_step(self,
                   real_batch,
                   netG,
                   optD,
                   log_data,
                   device=None,
                   global_step=None,
                   **kwargs):
        r"""
        Takes one training step for D.

        Args:
            real_batch (Tensor): A batch of real signals of shape (N, C, L).
            netG (nn.Module): Generator model for obtaining fake signals.
            optD (Optimizer): Optimizer for updating discriminator's parameters.
            device (torch.device): Device to use for running the model.
            log_data (MetricLog): An object to add custom metrics for visualisations.
            global_step (int): Variable to sync training, logging and checkpointing.
                Useful for dynamic changes to model amidst training.

        Returns:
            MetricLog: Returns MetricLog object containing updated logging variables after 1 training step.

        """
        self.zero_grad()

        # Produce real signals
        real_signals = real_batch
        batch_size = real_signals.shape[0]  # Match batch sizes for last iter

        # Produce fake signals
        fake_signals = netG.generate_signals(num_signals=batch_size, device=device).detach()

        # Produce logits for real and fake signals
        output_real = self.forward(real_signals)
        output_fake = self.forward(fake_signals)

        # Compute losses
        errD = self.compute_gan_loss(output_real=output_real,
                                     output_fake=output_fake)

        errD_GP = self.compute_gradient_penalty_loss(real_signals=real_signals,
                                                     fake_signals=fake_signals,
                                                     gp_scale=self.gp_scale)

        # Backprop and update gradients
        errD_total = errD + errD_GP
        errD_total.backward()
        optD.step()

        # Compute probabilities
        D_x, D_Gz = self.compute_probs(output_real=output_real,
                                       output_fake=output_fake)

        log_data.add_metric('errD', errD, group='loss')
        log_data.add_metric('D(x)', D_x, group='prob')
        log_data.add_metric('D(G(z))', D_Gz, group='prob')

        return log_data

    def compute_gradient_penalty_loss(self,
                                      real_signals,
                                      fake_signals,
                                      gp_scale=10.0):
        r"""
        Computes gradient penalty loss, as based on:
        https://github.com/jalola/improved-wgan-pytorch/blob/master/gan_train.py

        Args:
            real_signals (Tensor): A batch of real signals of shape (N, 3, L). // TODO: make num of channels configurable
            fake_signals (Tensor): A batch of fake signals of shape (N, 3, L).
            gp_scale (float): Gradient penalty lamda parameter.

        Returns:
            Tensor: Scalar gradient penalty loss.
        """
        # Obtain parameters
        N, _, L = real_signals.shape
        device = real_signals.device

        # Randomly sample some alpha between 0 and 1 for interpolation
        # where alpha is of the same shape for elementwise multiplication.
        alpha = torch.rand(N, 1)
        alpha = alpha.expand(N, int(real_signals.nelement() / N)).contiguous()
        alpha = alpha.view(N, 1, L)
        alpha = alpha.to(device)

        # Obtain interpolates on line between real/fake signals.
        interpolates = alpha * real_signals.detach() \
            + ((1 - alpha) * fake_signals.detach())
        interpolates = interpolates.to(device)
        interpolates.requires_grad_(True)

        # Get gradients of interpolates
        disc_interpolates = self.forward(interpolates)
        gradients = autograd.grad(outputs=disc_interpolates,
                                  inputs=interpolates,
                                  grad_outputs=torch.ones(
                                      disc_interpolates.size()).to(device),
                                  create_graph=True,
                                  retain_graph=True,
                                  only_inputs=True)[0]
        gradients = gradients.view(gradients.size(0), -1)

        # Compute GP loss
        gradient_penalty = (
            (gradients.norm(2, dim=1) - 1)**2).mean() * gp_scale

        return gradient_penalty
    

class Generator(WGANGPBaseGenerator):
    r"""
    Base class for a generic unconditional generator model.
    ResNet backbone generator for SNGAN.

    Attributes:
        nz (int): Noise dimension for upsampling.
        ngf (int): Variable controlling generator feature map sizes.
        bottom_width (int): Starting width for upsampling generator output to an image.
        loss_type (str): Name of loss to use for GAN loss.
    """
    def __init__(self, **kwargs):
        super().__init__(nz=500, ngf=150, sequence_length=54)
        self.errG_array = []
        self.count = 0

        # Build the layers
        self.l1 = nn.Linear(self.nz, self.sequence_length * self.ngf)
        self.block1 = GBlock(self.ngf, self.ngf, upsample=True)
        self.block2 = GBlock(self.ngf, self.ngf, upsample=False)
        self.block3 = GBlock(self.ngf, self.ngf, upsample=True)
        self.block4 = GBlock(self.ngf, self.ngf, upsample=False)
        self.block5 = GBlock(self.ngf, self.ngf, upsample=True)
        self.block6 = GBlock(self.ngf, self.ngf, upsample=False)
        self.block7 = GBlock(self.ngf, self.ngf, upsample=True)
        self.block8 = GBlock(self.ngf, self.ngf, upsample=False)
        self.block9 = GBlock(self.ngf, self.ngf, upsample=True)
        self.block10 = GBlock(self.ngf, self.ngf, upsample=False)
        self.block11 = GBlock(self.ngf, self.ngf, upsample=True)
        self.block12 = GBlock(self.ngf, self.ngf, upsample=False)
        self.c13 = nn.Conv1d(self.ngf, 64, 1, 1, 0)
        self.end = nn.Linear(3204, 3152)

        # Initialise the weights
        nn.init.normal_(self.l1.weight.data, 0.0, 0.02)
        nn.init.normal_(self.c13.weight.data, 0.0, 0.02)
        nn.init.normal_(self.end.weight.data, 0.0, 0.02)

    def forward(self, x):
        r"""
        Feedforwards a batch of noise vectors into a batch of fake images.

        Args:
            x (Tensor): A batch of noise vectors of shape (N, nz).

        Returns:
            Tensor: A batch of fake images of shape (N, C, L).
        """
        h = self.l1(x)
        h = h.view(x.shape[0], self.ngf, self.sequence_length)
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.block5(h)
        h = self.block6(h)
        h = self.block7(h)
        h = self.block8(h)
        h = self.block9(h)
        h = self.block10(h)
        h = self.block11(h)
        h = self.block12(h)
        h = self.c13(h)
        h = self.end(h)
        return h

    def generate_signals(self, num_signals, device=None):
        r"""
        Generates num_signals randomly.

        Args:
            num_signals (int): Number of signals to generate
            device (torch.device): Device to send images to.

        Returns:
            Tensor: A batch of generated images.
        """
        if device is None:
            device = self.device

        noise = torch.randn((num_signals, self.nz), device=device)
        fake_signals = self.forward(noise)

        return fake_signals

    def train_step(self,
                   real_batch,
                   netD,
                   optG,
                   log_data,
                   device=None,
                   global_step=None,
                   **kwargs):
        r"""
        Takes one training step for G.

        Args:
            real_batch (Tensor): A batch of real images of shape (N, C, L).
                Used for obtaining current batch size.
            netD (nn.Module): Discriminator model for obtaining losses.
            optG (Optimizer): Optimizer for updating generator's parameters.
            log_data (dict): A dict mapping name to values for logging uses.
            device (torch.device): Device to use for running the model.
            global_step (int): Variable to sync training, logging and checkpointing.
                Useful for dynamic changes to model amidst training.

        Returns:
            Returns MetricLog object containing updated logging variables after 1 training step.

        """
        batch_size = real_batch[0].shape[0]  # Get only batch size from real batch

        # Produce logits for fake signals
        fake_signals = self.generate_signals(num_signals=batch_size, device=device)
        out_fake = netD(fake_signals)

        self.zero_grad()

        # Backprop and update gradients
        errG = self.compute_gan_loss(out_fake)
        errG.backward()
        optG.step()

        # Log statistics
        self.errG_array.append(errG.item())
        if (self.count != 0 and self.count % 100 == 0):
            print(errG.item(), 'gen')
        self.count += 1
        log_data.add_metric('errG', errG.item(), group='loss')
        return log_data



class Discriminator(WGANGPBaseDiscriminator):
    def __init__(self, **kwargs):
        super().__init__(ndf=150)
        self.count = 0
        self.errD_array = []

        self.sblock1 = DBlock(64, self.ndf, kernel_size=3, stride=1, padding=0, downsample=False, reflectionPad=True)
        self.sblock2 = DBlock(self.ndf, self.ndf, kernel_size=3, stride=1, padding=0, downsample=True, reflectionPad=True)
        self.sblock3 = DBlock(self.ndf, self.ndf, kernel_size=3, stride=1, padding=0, downsample=False, reflectionPad=True)
        self.sblock4 = DBlock(self.ndf, self.ndf, kernel_size=3, stride=1, padding=0, downsample=True, reflectionPad=True)
        self.sblock5 = DBlock(self.ndf, self.ndf, kernel_size=3, stride=1, padding=0, downsample=False, reflectionPad=True)
        self.sblock6 = DBlock(self.ndf, self.ndf, kernel_size=3, stride=1, padding=0, downsample=True, reflectionPad=True)
        self.sblock7 = DBlock(self.ndf, self.ndf, kernel_size=3, stride=1, padding=0, downsample=False, reflectionPad=True)
        self.sblock8 = DBlock(self.ndf, self.ndf, kernel_size=3, stride=1, padding=0, downsample=True, reflectionPad=True)
        self.sblock9 = DBlock(self.ndf, self.ndf, kernel_size=3, stride=1, padding=0, downsample=False, reflectionPad=True)
        self.sblock10 = DBlock(self.ndf, self.ndf, kernel_size=3, stride=1, padding=0, downsample=True, reflectionPad=True)
        self.sblock11 = DBlock(self.ndf, self.ndf, kernel_size=3, stride=1, padding=0, downsample=False, reflectionPad=True)
        self.sblock12 = DBlock(self.ndf, self.ndf, kernel_size=3, stride=1, padding=0, downsample=True, reflectionPad=True)
        self.c = nn.Conv1d(self.ndf, 64, 1, 1, 0)
        self.end = nn.Linear(45, 1)
        
        nn.init.normal_(self.c.weight.data, 0.0, 0.02)
        nn.init.normal_(self.end.weight.data, 0.0, 0.02)

    def forward(self, x):
        x = x.float()
        h = self.sblock1(x)
        h = self.sblock2(h)
        h = self.sblock3(h)
        h = self.sblock4(h)
        h = self.sblock5(h)
        h = self.sblock6(h)
        h = self.sblock7(h)
        h = self.sblock8(h)
        h = self.sblock9(h)
        h = self.sblock10(h)
        h = self.sblock11(h)
        h = self.sblock12(h)
        h = self.c(h)
        h = self.end(h)
        return h.view(h.shape[0], 64)

    def train_step(self,
                   real_batch,
                   netG,
                   optD,
                   log_data,
                   device=None,
                   global_step=None,
                   **kwargs):
        r"""
        Takes one training step for D.

        Args:
            real_batch (Tensor): A batch of real images of shape (N, C, H, W).
            loss_type (str): Name of loss to use for GAN loss.
            netG (nn.Module): Generator model for obtaining fake images.
            optD (Optimizer): Optimizer for updating discriminator's parameters.
            device (torch.device): Device to use for running the model.
            log_data (dict): A dict mapping name to values for logging uses.
            global_step (int): Variable to sync training, logging and checkpointing.
                Useful for dynamic changes to model amidst training.

        Returns:
            MetricLog: Returns MetricLog object containing updated logging variables after 1 training step.
        """
        batch_size = real_batch.shape[0] # Match batch sizes for last iter

        # Produce logits for real signals
        real_signals = real_batch
        out_real = self.forward(real_signals)

        # Produce logits for fake signals
        fake_signals = netG.generate_signals(num_signals=batch_size, device=device).detach()
        out_fake = self.forward(fake_signals)

        # Reset the gradients to zero
        optD.zero_grad()

        # Backprop and update gradients
        errD = self.compute_gan_loss(output_real=out_real, output_fake=out_fake)
        errD_GP = self.compute_gradient_penalty_loss(real_signals=real_signals, fake_signals=fake_signals)

        errD_total = errD + errD_GP
        errD_total.backward()
        optD.step()


        # Log statistics
        if (self.count != 0 and self.count % 5 == 0):
            self.errD_array.append(errD_total.item())
            log_data.add_metric('errD', errD.item(), group='loss')
            log_data.add_metric('errD_GP', errD_GP.item(), group='loss')
        if (self.count != 0 and self.count % 500 == 0):
            print(errD_total.item(), 'disc')
        if (self.count % 5000 == 0):
            plot_everything(fake_signals, netG.errG_array, self.errD_array, 
                            self.count, show_result=False, save_result=True)
        self.count += 1

        return log_data

    def compute_gradient_penalty_loss(self,
                                      real_signals,
                                      fake_signals,
                                      gp_scale=10.0):
        r"""
        Computes gradient penalty loss, as based on:
        https://github.com/jalola/improved-wgan-pytorch/blob/master/gan_train.py

        Args:
            real_signals (Tensor): A batch of real signals of shape (N, 1, L). // TODO: make num of channels configurable
            fake_signals (Tensor): A batch of fake signals of shape (N, 1, L).
            gp_scale (float): Gradient penalty lamda parameter.

        Returns:
            Tensor: Scalar gradient penalty loss.
        """
        # Obtain parameters
        N, _, L = real_signals.shape
        device = real_signals.device

        # Randomly sample some alpha between 0 and 1 for interpolation
        # where alpha is of the same shape for elementwise multiplication.
        alpha = torch.rand(N, 1)
        alpha = alpha.expand(N, int(real_signals.nelement() / N)).contiguous()
        alpha = alpha.view(N, 64, L)  # TODO: MAKE CHANNEL VARIABLE
        alpha = alpha.to(device)

        # Obtain interpolates on line between real/fake signals.
        interpolates = alpha * real_signals.detach() \
            + ((1 - alpha) * fake_signals.detach())
        interpolates = interpolates.to(device)
        interpolates.requires_grad_(True)

        # Get gradients of interpolates
        disc_interpolates = self.forward(interpolates)
        gradients = autograd.grad(outputs=disc_interpolates,
                                  inputs=interpolates,
                                  grad_outputs=torch.ones(
                                      disc_interpolates.size()).to(device),
                                  create_graph=True,
                                  retain_graph=True,
                                  only_inputs=True)[0]
        gradients = gradients.view(gradients.size(0), -1)

        # Compute GP loss
        gradient_penalty = (
            (gradients.norm(2, dim=1) - 1)**2).mean() * gp_scale

        return gradient_penalty