"""This module contains simple helper functions """
from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os


def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.
    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = (
            image_tensor.cpu().float().numpy()
        )  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (
            (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
        )  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)


def diagnose_network(net, name="network"):
    """Calculate and print the mean of average absolute(gradients)
    Parameters:
        net (torch network) -- Torch network
        name (str) -- the name of the network
    """
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path, aspect_ratio=1.0):
    """Save a numpy image to the disk
    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """

    image_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape

    if aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    if aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array
    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print("shape,", x.shape)
    if val:
        x = x.flatten()
        print(
            "mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f"
            % (np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x))
        )


def mkdirs(paths):
    """create empty directories if they don't exist
    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist
    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)


def decode_md(img):
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()

    if len(img.shape) == 2:
        img = np.expand_dims(img, 0)

    if len(img.shape) == 3:
        img = [img]

    ims = []
    for im in img:
        ims.append(1 / np.exp(2 * im + 2))

    if len(ims[0].shape) == 3:
        ims = [np.expand_dims(im, 0) for im in ims]

    return np.concatenate(ims, axis=0)


ref_one_hot = {
    0: [1, 0, 0, 0],
    90: [0, 1, 0, 0],
    180: [0, 0, 1, 0],
    270: [0, 0, 0, 1],
}
ref = {k: int(np.argmax(v)) for k, v in ref_one_hot.items()}
labref_one_hot = {
    0: [1, 0, 0, 0],
    1: [0, 1, 0, 0],
    2: [0, 0, 1, 0],
    3: [0, 0, 0, 1],
}
labref = {i: i for i in range(4)}


def angle_to_tensor(angle, one_hot=False):
    global ref
    global ref_one_hot
    global labref
    global labref_one_hot
    angle = int(angle)
    if angle in ref:
        r = ref_one_hot if one_hot else ref
    else:
        if angle in labref:
            r = labref_one_hot if one_hot else labref
        else:
            raise ValueError("Unknown angle {}".format(angle))

    return torch.tensor(r[angle])


def angles_to_tensors(angles, one_hot=False):
    return torch.cat([angle_to_tensor(a, one_hot).unsqueeze(0) for a in angles], dim=0)


def env_to_path(path_str):
    """Transorms an environment variable mention in a json
    into its actual value. E.g. $HOME/clouds -> /home/vsch/clouds

    Args:
        path_str (str): path_str potentially containing the env variable

    """
    if not path_str:
        return path_str

    path_elements = path_str.split("/")
    new_path = []
    for el in path_elements:
        if "$" in el:
            new_path.append(os.environ[el.replace("$", "")])
        else:
            new_path.append(el)
    return "/".join(new_path)
