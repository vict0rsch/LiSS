import comet_ml
import sys
import time
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from unittest.mock import MagicMock


sys.path.append(str(Path(__file__).resolve().parent.parent))
from options.train_options import TrainOptions
from models import create_model
from data import create_dataset
from copy import copy
from eval import eval


class MockExp:
    def __init__(self, *args, **kwargs):
        pass

    def __getattribute__(self, attr):
        return MagicMock(return_value=False)


def show_tensor(t):
    t = t.detach().cpu().numpy()
    t -= t.min()
    t /= t.max()
    t = np.transpose(t, (1, 2, 0))
    plt.imshow(t)
    plt.show(block=True)


if __name__ == "__main__":

    # local: run test.py --dataroot=../s2w_d --gpu_ids -1 --netG liss
    # beluga: python test.py --dataroot=$SLURM_TMPDIR/s2w_d --netG liss --batch_size=5

    opt = TrainOptions().parse()
    dataset = create_dataset(opt)
    test_opt = copy(opt)
    test_opt.serial_batches = True
    test_opt.phase = "test"
    test_dataset = create_dataset(test_opt)
    if opt.model == "liss":
        opt.netD = "rotational"

    exp = MockExp()

    g_step = True
    eval_model = True
    test_metrics = True
    show_rot = True
    full_step = True
    merge_encoders = True

    if show_rot:
        b = next(iter(dataset))
        c = False
        i = 0
        while c:
            print(b["angleA"][i])
            show_tensor(b["rA"][i])
            c = input("Next Image? (y)") == "y"
            i += 1
        model = create_model(opt)
        model.set_input(b)
        model.forward()

    if full_step:
        model = create_model(opt)
        model.exp = MockExp()
        b = next(iter(dataset))
        stime = time.time()
        model.set_input(b)
        model.optimize_parameters()
        print("Update time: {:.3f}".format(time.time() - stime))
        for d in dir(model):
            if d.startswith("loss_") and isinstance(model.get(d), torch.Tensor):
                print("{:30}: {}".format(d, model.get(d).item()))

    if g_step:
        model = create_model(opt)
        model.exp = MockExp()
        G = model.netG_A
        if isinstance(G, torch.nn.DataParallel):
            e = G.module.encoder
            d = G.module.depth
            t = G.module.decoder
            r = G.module.rotation
        else:
            e = G.encoder
            d = G.depth
            t = G.decoder
            r = G.rotation

        x = torch.rand(1, 3, 256, 256).to(next(G.parameters()).device)
        z = e(x)
        b = next(iter(dataset))

        print("Setting input")
        model.set_input(b)
        print("Forward")
        model.forward()
        print("Setting requires_grad : False to netDs")
        model.set_requires_grad(
            [model.netD_A, model.netD_B], False
        )  # Ds require no gradients when optimizing Gs
        print("Backward")
        model.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        model.backward_G()  # calculate gradients for G_A and G_B
        for d in dir(model):
            if d.startswith("loss_") and "names" not in d:
                print("{:15} {:.3f}".format(d, getattr(model, d).item()))
        print("Step")
        model.optimizer_G.step()  # update G_A and G_B's weights

    if eval_model:
        model = create_model(opt)
        exp = comet_ml.Experiment(project_name="LiSS")
        exp.add_tag(Path(opt.dataroot).name)
        exp.add_tag(opt.model)
        model.exp = exp
        if "task_schedule" in opt:
            exp.add_tag(opt.task_schedule)
        exp.add_tag("functional_test")
        metrics = eval(model, test_dataset, exp, 1234)

    if merge_encoders:
        model = create_model(opt)
        model.exp = MockExp()
        encA = model.netG_A.module.get_encoder()
        encA = nn.DataParallel(encA.to(0), [0])
        encB = model.netG_B.module.get_encoder()
        encB = nn.DataParallel(encB.to(0), [0])

        print(encA.module.model[1].weight[0, 0, 0])
        print(encB.module.model[1].weight[0, 0, 0])

        alpha = 0.5
        encB.load_state_dict(
            {
                k: alpha * v1 + (1 - alpha) * v2
                for (k, v1), (_, v2) in zip(
                    encA.state_dict().items(), encB.state_dict().items()
                )
            }
        )
        print(encB.module.model[1].weight[0, 0, 0])

    if test_metrics:
        opt.task_schedule = "liss"

        def print_computes(model):
            for d in dir(model):
                if d.startswith("_should_compute"):
                    print("{:30}: {}".format(d, model.get(d)))
            print("\n\n")

        model = create_model(opt)
        model.exp = MockExp()
        model.setup(opt)
        metrics = {
            "test_G_A_rotation_acc": 0,
            "test_G_B_rotation_acc": 0,
            "test_G_A_jigsaw_acc": 0,
            "test_G_B_jigsaw_acc": 0,
            "test_G_A_depth_loss": 10,
            "test_G_B_depth_loss": 10,
            "test_G_A_gray_loss": 10,
            "test_G_B_gray_loss": 10,
        }
        print("--------------------")
        model.update_task_schedule(metrics)
        print_computes(model)

        print("--------------------")
        metrics["test_G_A_rotation_acc"] = metrics["test_G_B_rotation_acc"] = 1
        model.update_task_schedule(metrics)
        print_computes(model)

        print("--------------------")
        metrics["test_G_A_jigsaw_acc"] = metrics["test_G_B_jigsaw_acc"] = 1
        model.update_task_schedule(metrics)
        print_computes(model)

        print("--------------------")
        metrics["test_G_A_depth_loss"] = metrics["test_G_B_depth_loss"] = 0
        metrics["test_G_A_jigsaw_acc"] = metrics["test_G_B_jigsaw_acc"] = 0
        model.update_task_schedule(metrics)
        print_computes(model)

        print("--------------------")
        metrics["test_G_A_gray_loss"] = metrics["test_G_B_gray_loss"] = 0
        model.update_task_schedule(metrics)
        print_computes(model)
