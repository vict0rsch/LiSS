from comet_ml import Experiment
import numpy as np
import torch
from models.liss_model import LiSSModel
from data.unaligned_dataset import UnalignedDataset
from copy import deepcopy


def swap_domain(domain):
    assert domain in {"A", "B"}
    if domain == "A":
        return "B"
    return "A"


def eval(
    model: LiSSModel,
    dataset: UnalignedDataset,
    exp: Experiment,
    total_iters: int = 0,
    nb_ims: int = 30,
):
    liss = model.opt.model == "liss"
    metrics = {}
    print(f"----------- Evaluation {total_iters} ----------")
    with torch.no_grad():
        data = {
            "translation": {
                "A": {"rec": None, "idt": None, "real": None, "fake": None},
                "B": {"rec": None, "idt": None, "real": None, "fake": None},
            }
        }
        force = set(["identity", "translation"])
        if liss:
            for t in model.tasks:
                tmp = {}
                if t.eval_visuals_pred or t.log_type == "acc":
                    tmp["pred"] = None
                if t.eval_visuals_target or t.log_type == "acc":
                    tmp["target"] = None
                data[t.key] = {domain: deepcopy(tmp) for domain in "AB"}

            force |= set(model.tasks.keys)

        losses = {
            k: []
            for k in dir(model)
            if k.startswith("loss_") and isinstance(getattr(model, k), torch.Tensor)
        }
        for i, b in enumerate(dataset):
            # print(f"\rEval batch {i}", end="")

            model.set_input(b)
            model.forward(force=force)
            model.backward_G(losses_only=True, force=force)

            for k in dir(model):
                if k.startswith("loss_") and isinstance(
                    getattr(model, k), torch.Tensor
                ):
                    if k not in losses:
                        losses[k] = []
                    losses[k].append(getattr(model, k).detach().cpu().item())

            if liss:

                for t in model.tasks:
                    for domain in "AB":
                        for dtype in data[t.key][domain]:
                            if (
                                t.log_type != "acc"
                                and data[t.key][domain][dtype] is not None
                                and len(data[t.key][domain][dtype]) >= nb_ims
                            ):
                                continue

                            v = model.get(f"{domain}_{t.key}_{dtype}").detach().cpu()
                            if data[t.key][domain][dtype] is None:
                                data[t.key][domain][dtype] = v
                            else:
                                data[t.key][domain][dtype] = torch.cat(
                                    [data[t.key][domain][dtype], v], dim=0,
                                )

            # -------------------------
            # -----  Translation  -----
            # -------------------------
            if (
                data["translation"]["A"]["real"] is None
                or len(data["translation"]["A"]["real"]) < nb_ims
            ):
                for domain in "AB":
                    for dtype in ["real", "fake", "rec", "idt"]:
                        dom = domain
                        if dtype in {"fake", "idt"}:
                            dom = swap_domain(domain)
                        v = model.get(f"{dom}_{dtype}").detach().cpu()
                        if data["translation"][domain][dtype] is None:
                            data["translation"][domain][dtype] = v
                        else:
                            data["translation"][domain][dtype] = torch.cat(
                                [data["translation"][domain][dtype], v], dim=0
                            )
                        # print(
                        #     f"{domain} {dtype} {len(data['translation'][domain][dtype])}"
                        # )

    for task in data:
        if task != "translation" and model.tasks[task].log_type != "vis":
            continue
        for domain in data[task]:
            for i, v in data[task][domain].items():
                data[task][domain][i] = torch.cat(
                    list(v[:nb_ims].permute(0, 2, 3, 1)), axis=1
                )

    log_images = int(
        data["translation"]["A"]["real"].shape[1]
        / data["translation"]["A"]["real"].shape[0]
    )
    im_size = data["translation"]["A"]["real"].shape[0]

    ims = {"A": None, "B": None}

    data_keys = ["translation"]
    translation_keys = ["real", "fake", "rec", "idt"]
    data_keys += [task for task in data if task not in data_keys]

    for task in data_keys:
        if task != "translation" and model.tasks[task].log_type != "vis":
            continue
        for domain in "AB":
            im_types = (
                translation_keys
                if task == "translation"
                else list(data[task][domain].keys())
            )
            for im_type in im_types:
                v = data[task][domain][im_type].float()
                if task == "depth":
                    v = to_min1_1(v)
                    v = v.repeat((1, 1, 3))
                v = v + 1
                v = v / 2
                if ims[domain] is None:
                    ims[domain] = v
                else:
                    ims[domain] = torch.cat([ims[domain], v], dim=0)

    # ------------------------
    # -----  Comet Logs  -----
    # ------------------------
    for i in range(0, log_images, 5):
        k = i + 5
        exp.log_image(
            ims["A"][:, i * im_size : k * im_size, :].numpy(),
            "test_A_{}_{}_rfcidg".format(i * 5, (i + 1) * 5 - 1),
            step=total_iters,
        )
        exp.log_image(
            ims["B"][:, i * im_size : k * im_size, :].numpy(),
            "test_B_{}_{}_rfcidg".format(i * 5, (i + 1) * 5 - 1),
            step=total_iters,
        )
    if liss:
        test_losses = {
            "test_" + ln: np.mean(losses["loss_" + ln])
            for t in model.tasks
            for ln in t.loss_names
        }

        test_accs = {
            f"test_G_{domain}_{t.key}_acc": np.mean(
                data[t.key][domain]["pred"].max(-1)[1].numpy()
                == data[t.key][domain]["target"].numpy()
            )
            for domain in "AB"
            for t in model.tasks
            if t.log_type == "acc"
        }

        if liss:
            exp.log_metrics(test_losses, step=total_iters)
            exp.log_metrics(test_accs, step=total_iters)

            for t in model.tasks:
                if t.log_type != "acc":
                    continue
                for domain in "AB":
                    target = data[t.key][domain]["target"].numpy()
                    pred = data[t.key][domain]["pred"].numpy()
                    exp.log_confusion_matrix(
                        get_one_hot(target, t.output_dim),
                        pred,
                        file_name=f"confusion_{domain}_{t.key}_{total_iters}.json",
                        title=f"confusion_{domain}_{t.key}_{total_iters}.json",
                    )
            metrics = {k + "_loss": v for k, v in test_losses.items()}
            metrics.update(test_accs)
    print("----------- End Evaluation----------")
    return metrics


def get_one_hot(targets, nb_classes):
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape) + [nb_classes])


def to_min1_1(im):
    im -= im.min()
    im /= im.max()
    im -= 0.5
    im *= 2
    return im
