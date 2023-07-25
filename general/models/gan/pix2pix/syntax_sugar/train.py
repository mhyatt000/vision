"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""

import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer

import time

def timeit(func):
    def wrapper(*args, **kwargs):
        start= time.time()
        result = func(*args, **kwargs)
        end= time.time()
        execution= end- start
        print(f"Function '{func.__name__}' took {execution} seconds to execute.")
        return result
    return wrapper


@timeit()
def step():
    """docstring"""

    total_iters += opt.batch_size
    epoch_iter += opt.batch_size

    model.set_input(data)
    model.optimize_parameters()  # calculate loss functions, get gradients, update network weights

    if total_iters % opt.display_freq == 0:
        save_result = total_iters % opt.update_html_freq == 0
        model.compute_visuals()
        visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

    if total_iters % opt.print_freq == 0:
        losses = model.get_current_losses()
        visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data) # time comp time data
        if opt.display_id > 0:
            visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

    if total_iters % opt.save_latest_freq == 0:
        print(f"saving the latest model (epoch {epoch}, total_iters {total_iters})")
        save_suffix = "iter_%d" % total_iters if opt.save_by_iter else "latest"
        model.save_networks(save_suffix)


@timeit()
def epoch():
    """docstring"""

    epoch_iter = 0
    visualizer.reset()  # reset the visualizer: make sure it saves the results to HTML at least once every epoch
    model.update_learning_rate()

    for i, data in enumerate(dataset):
        step()

    if epoch % opt.save_epoch_freq == 0:
        print(f"saving the model at the end of epoch {epoch}, iters {total_iters})")
        model.save_networks("latest")
        model.save_networks(epoch)


def main():
    """docstring"""

    opt = TrainOptions().parse()
    dataset = create_dataset(opt)
    dataset_size = len(dataset)
    print(f"The number of training images = {dataset_size}")

    model = create_model(opt)
    model.setup(opt)  # regular setup: load and print networks; create schedulers

    visualizer = Visualizer(opt)  # create a visualizer that display/save images and plots
    total_iters = 0

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        epoch()


if __name__ == "__main__":
    main()
