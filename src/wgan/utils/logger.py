# Author: Joshua Park
# Modified by: Akira Kudo
# Created: 2024/11/01
# Last Updated: 2024/11/18

"""
Implementation of the Logger object for performing training logging and visualisation.
"""
import os

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision import utils as vutils


class Logger:
    """
    Writes summaries and visualises training progress.

    Attributes:
        log_dir (str): The path to store logging information.
        num_steps (int): Total number of training iterations.
        dataset_size (int): The number of examples in the dataset.
        device (Device): Torch device object to send data to.
        flush_secs (int): Number of seconds before flushing summaries to disk.
        writers (dict): A dictionary of tensorboard writers with keys as metric names.
        num_epochs (int): The number of epochs, for extra information.
    """
    def __init__(self,
                 log_dir,
                 num_steps,
                 dataset_size,
                 device,
                 flush_secs=120,
                 **kwargs):
        self.log_dir = log_dir
        self.num_steps = num_steps
        self.dataset_size = dataset_size
        self.flush_secs = flush_secs
        self.num_epochs = self._get_epoch(num_steps)
        self.device = device
        self.writers = {}

        # Create log directory if haven't already
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

    def _get_epoch(self, steps):
        """
        Helper function for getting epoch.
        """
        return max(int(steps / self.dataset_size), 1)

    def _build_writer(self, metric):
        writer = SummaryWriter(log_dir=os.path.join(self.log_dir, 'data',
                                                    metric),
                               flush_secs=self.flush_secs)

        return writer

    def write_summaries(self, log_data, global_step):
        """
        Tasks appropriate writers to write the summaries in tensorboard. Creates additional
        writers for summary writing if there are new scalars to log in log_data.

        Args:
            log_data (MetricLog): Dict-like object to collect log data for TB writing.
            global_step (int): Global step variable for syncing logs.

        Returns:
            None
        """
        for metric, data in log_data.items():
            if metric not in self.writers:
                self.writers[metric] = self._build_writer(metric)

            # Write with a group name if it exists
            name = log_data.get_group_name(metric) or metric
            self.writers[metric].add_scalar(name,
                                            log_data[metric],
                                            global_step=global_step)

    def close_writers(self):
        """
        Closes all writers.
        """
        for metric in self.writers:
            self.writers[metric].close()

    def print_log(self, global_step, log_data, time_taken):
        """
        Formats the string to print to stdout based on training information.

        Args:
            log_data (MetricLog): Dict-like object to collect log data for TB writing.
            global_step (int): Global step variable for syncing logs.
            time_taken (float): Time taken for one training iteration.

        Returns:
            str: String to be printed to stdout.
        """
        # Basic information
        log_to_show = [
            "INFO: [Epoch {:d}/{:d}][Global Step: {:d}/{:d}]".format(
                self._get_epoch(global_step), self.num_epochs, global_step,
                self.num_steps)
        ]

        # Display GAN information as fed from user.
        GAN_info = [""]
        metrics = sorted(log_data.keys())

        for metric in metrics:
            GAN_info.append('{}: {}'.format(metric, log_data[metric]))

        # Add train step time information
        GAN_info.append("({:.4f} sec/idx)".format(time_taken))

        # Accumulate to log
        log_to_show.append("\n| ".join(GAN_info))

        # Finally print the output
        ret = " ".join(log_to_show)
        print(ret)

        return ret

    def _get_fixed_noise(self, nz, num_signals, output_dir=None):
        """
        Produce the fixed gaussian noise vectors used across all models
        for consistency.
        """
        if output_dir is None:
            output_dir = os.path.join(self.log_dir, 'viz')

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_file = os.path.join(output_dir,
                                   'fixed_noise_nz_{}.pth'.format(nz))

        if os.path.exists(output_file):
            noise = torch.load(output_file)

        else:
            noise = torch.randn((num_signals, nz))
            torch.save(noise, output_file)

        return noise.to(self.device)

    def _get_fixed_labels(self, num_signals, num_classes):
        """
        Produces fixed class labels for generating fixed signals.
        """
        labels = np.array([i % num_classes for i in range(num_signals)])
        labels = torch.from_numpy(labels).to(self.device)

        return labels

    def vis_signals(self, netG, global_step, num_signals=64):
        """
        Produce visualisations of the G(z), one fixed and one random.

        Args:
            netG (Module): Generator model object for producing signals.
            global_step (int): Global step variable for syncing logs.
            num_signals (int): The number of signals to visualise.

        Returns:
            None
        """
        img_dir = os.path.join(self.log_dir, 'signals')
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)

        with torch.no_grad():
            # Generate random signals
            noise = torch.randn((num_signals, netG.nz), device=self.device)
            fake_signals = netG(noise).detach().cpu()

            # Generate fixed random signals
            fixed_noise = self._get_fixed_noise(nz=netG.nz,
                                                num_signals=num_signals)

            if hasattr(netG, 'num_classes') and netG.num_classes > 0:
                fixed_labels = self._get_fixed_labels(num_signals,
                                                      netG.num_classes)
                fixed_fake_signals = netG(fixed_noise,
                                         fixed_labels).detach().cpu()
            else:
                fixed_fake_signals = netG(fixed_noise).detach().cpu()

            # Map name to results
            signals_dict = {
                'fixed_fake': fixed_fake_signals,
                'fake': fake_signals
            }

            # Visualise all results
            for name, signals in signals_dict.items():
                signals_viz = vutils.make_grid(signals,
                                              padding=2,
                                              normalize=True)

                vutils.save_signal(signals_viz,
                                  '{}/{}_samples_step_{}.png'.format(
                                      img_dir, name, global_step),
                                  normalize=True)

                if 'img' not in self.writers:
                    self.writers['img'] = self._build_writer('img')

                self.writers['img'].add_signal('{}_vis'.format(name),
                                              signals_viz,
                                              global_step=global_step)
"""
MetricLog object for intelligently logging data to display them more intuitively.
"""


class MetricLog:
    """
    A dictionary-like object that logs data, and includes an extra dict to map the metrics
    to its group name, if any, and the corresponding precision to print out.

    Attributes:
        metrics_dict (dict): A dictionary mapping to another dict containing
            the corresponding value, precision, and the group this metric belongs to.
    """
    def __init__(self, **kwargs):
        self.metrics_dict = {}

    def add_metric(self, name, value, group=None, precision=4):
        """
        Logs metric to internal dict, but with an additional option
        of grouping certain metrics together.

        Args:
            name (str): Name of metric to log.
            value (Tensor/Float): Value of the metric to log.
            group (str): Name of the group to classify different metrics together.
            precision (int): The number of floating point precision to represent the value.

        Returns:
            None
        """
        # Grab tensor values only
        try:
            value = value.item()
        except AttributeError:
            value = value

        self.metrics_dict[name] = dict(value=value,
                                       group=group,
                                       precision=precision)

    def __getitem__(self, key):
        return round(self.metrics_dict[key]['value'],
                     self.metrics_dict[key]['precision'])

    def get_group_name(self, name):
        """
        Obtains the group name of a particular metric. For example, errD and errG
        which represents the discriminator/generator losses could fall under a
        group name called "loss".

        Args:
            name (str): The name of the metric to retrieve group name.

        Returns:
            str: A string representing the group name of the metric.
        """
        return self.metrics_dict[name]['group']

    def keys(self):
        """
        Dict like functionality for retrieving keys.
        """
        return self.metrics_dict.keys()

    def items(self):
        """
        Dict like functionality for retrieving items.
        """
        return self.metrics_dict.items()