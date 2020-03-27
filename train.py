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
import comet_ml
import numpy as np
import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from copy import copy
from eval import eval
from pathlib import Path
import os
from collections import deque

if __name__ == "__main__":

    # ---------------------
    # -----  Options  -----
    # ---------------------
    opt = TrainOptions().parse()  # get training options
    test_opt = copy(opt)
    test_opt.phase = "test"
    test_opt.serial_batches = True
    if opt.model == "liss":
        opt.netD = "rotational"  # for api ease ; rotations not used unless --rot_D

    # ----------------------
    # -----  Datasets  -----
    # ----------------------
    dataset = create_dataset(opt)
    test_dataset = create_dataset(test_opt)
    dataset_size = len(dataset)  # get the number of images in the dataset.
    print("The number of training images = %d" % dataset_size)

    # ------------------------------
    # -----  Comet Experiment  -----
    # ------------------------------
    exp = comet_ml.Experiment(project_name="LiSS", auto_metric_logging=False)
    exp.log_parameters(dict(vars(opt)))
    exp.add_tag(Path(opt.dataroot).name)
    exp.add_tag(opt.model)
    exp.log_parameter("slurm_job_id", os.environ.get("SLURM_JOB_ID", ""))
    if "message" in opt:
        exp.log_text(opt.message)
    if "task_schedule" in opt:
        exp.add_tag(opt.task_schedule)
    if "small_data" in opt and opt.small_data > 0:
        exp.add_tag("small")
    # if "D_rotation" in opt and opt.D_rotation:
    #     exp.add_tag("D_rot")

    # -------------------
    # -----  Model  -----
    # -------------------
    model = create_model(opt)  # create a model given opt.model and other options
    model.exp = exp
    model.setup(opt)  # regular setup: load and print networks; create schedulers

    if "radam" in model.optimizer_G.__class__.__name__.lower():
        exp.log_parameter("radam", True)

    # ------------------------
    # -----  Iterations  -----
    # ------------------------
    total_iters = 0  # the total number of training iterations
    iter_times = deque(maxlen=15)
    iter_times.append(1)
    print("starting")

    epoch = opt.epoch_count - 1
    decay_epochs = 0
    while epoch > -2:
        epoch += 1
        exp.log_parameter("curr_epoch", epoch)
        # outer loop for different epochs; we save the model by <epoch_count>,
        # <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()  # timer for data loading per iteration
        epoch_iter = 0  # the number of training iterations in current epoch
        print(f"--- epoch {epoch} starting")

        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            total_iters += opt.batch_size
            epoch_iter += opt.batch_size

            # ------------------------
            # -----  Train Step  -----
            # ------------------------
            # unpack data from dataset and apply preprocessing then
            # calculate loss functions, get gradients, update network weights
            model.total_iters = total_iters
            model.set_input(data)
            model.optimize_parameters()

            iter_times.append((time.time() - iter_start_time) / opt.batch_size)
            # ------------------------
            # -----  Validation  -----
            # ------------------------
            if total_iters == opt.batch_size or total_iters % opt.display_freq == 0:
                metrics = eval(model, test_dataset, exp, total_iters)
                model.update_task_schedule(metrics)

            # ------------------------------------------
            # -----  Logging and Saving utilities  -----
            # ------------------------------------------
            # print training losses and save logging information to the disk
            if total_iters % opt.print_freq == 0:
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                exp.log_metrics(losses, step=total_iters)
                exp.log_metric("sample_time", np.mean(iter_times))
                for d in dir(model):
                    if d.startswith("_should_compute") and isinstance(
                        model.get(d), bool
                    ):
                        exp.log_metric(d, int(model.get(d)))

            # cache our latest model every <save_latest_freq> iterations
            if total_iters % opt.save_latest_freq == 0:
                print(
                    "saving the latest model (epoch %d, total_iters %d)"
                    % (epoch, total_iters)
                )
                save_suffix = "iter_%d" % total_iters if opt.save_by_iter else "latest"
                model.save_networks(save_suffix)

            # stdout print
            if i % 50 == 0:
                print(
                    "Iter {} ({}) | {:.2f}\r".format(
                        i, total_iters, np.mean(iter_times)
                    ),
                    end="",
                )

            iter_data_time = time.time()
            # -------------------------
            # -----  END OF STEP  -----
            # -------------------------
        if epoch % opt.save_epoch_freq == 0:
            # cache our model every <save_epoch_freq> epochs
            print(
                "saving the model at the end of epoch %d, iters %d"
                % (epoch, total_iters)
            )
            model.save_networks("latest")
            model.save_networks(epoch)

        print(
            "End of epoch %d / %d \t Time Taken: %d sec"
            % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time)
        )
        # model.update_learning_rate()  # update learning rates at the end of every epoch.
        if epoch > opt.n_epochs:
            if opt.model == "liss":
                if model.should_compute("translation"):
                    decay_epochs += 1
            else:
                decay_epochs += 1
        # ----------------------------------------
        # -----  Linear Learning Rate Decay  -----
        # ----------------------------------------
        if decay_epochs > 0:
            for g in model.optimizer_G.param_groups:
                g["lr"] = (
                    g["lr"]
                    / (1 - (decay_epochs - 1) / (opt.n_epochs_decay + 1))
                    * (1 - decay_epochs / (opt.n_epochs_decay + 1))
                )
            for g in model.optimizer_D.param_groups:
                g["lr"] = (
                    g["lr"]
                    / (1 - (decay_epochs - 1) / (opt.n_epochs_decay + 1))
                    * (1 - decay_epochs / (opt.n_epochs_decay + 1))
                )

        lrs = {}
        for g in model.optimizer_G.param_groups:
            lrs["lr_" + g["group_name"]] = g["lr"]
        exp.log_metrics(lrs)

        if decay_epochs > opt.n_epochs_decay:
            epoch = -2
    # ---------------------
    # -----  THE END  -----
    # ---------------------
    exp.add_tag("finished")
