import os.path
from data.base_dataset import BaseDataset, get_transform, get_dic_transform
from data.image_folder import make_dataset
from PIL import Image
import random
from pathlib import Path
from copy import copy
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
sys.path.append(str(Path(__file__).resolve().parent.parent))
from models.task import AuxiliaryTasks


class UnalignedDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.
    It requires two directories to host training images
    from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.
        Parameters:
            opt (Option class) -- stores all the experiment flags;
                needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)

        if self.opt.netG == "liss":
            self.tasks = AuxiliaryTasks(
                [t.strip() for t in opt.auxiliary_tasks.split(",")]
            )
        self.load_depth = self.opt.netG == "liss"
        self.load_rotation = self.opt.netG == "liss"

        self.dir_A = os.path.join(
            opt.dataroot, opt.phase + "A"
        )  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(
            opt.dataroot, opt.phase + "B"
        )  # create a path '/path/to/data/trainB'

        self.A_paths = sorted(
            make_dataset(self.dir_A, opt.max_dataset_size)
        )  # load images from '/path/to/data/trainA'
        self.B_paths = sorted(
            make_dataset(self.dir_B, opt.max_dataset_size)
        )  # load images from '/path/to/data/trainB'
        if opt.small_data > 0:
            self.A_paths = self.A_paths[: opt.small_data]
            self.B_paths = self.B_paths[: opt.small_data]

        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        btoA = self.opt.direction == "BtoA"
        input_nc = (
            self.opt.output_nc if btoA else self.opt.input_nc
        )  # get the number of channels of input image
        output_nc = (
            self.opt.input_nc if btoA else self.opt.output_nc
        )  # get the number of channels of output image
        if opt.netG != "liss":
            self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1))
            self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))
        else:
            self.transform_A = get_dic_transform(
                self.opt, grayscale=False, tasks=self.tasks
            )
            self.transform_B = get_dic_transform(
                self.opt, grayscale=False, tasks=self.tasks
            )

    def __getitem__(self, index):
        """Return a data point and its metadata information.
        Parameters:
            index (int)      -- a random integer for data indexing
        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = self.A_paths[
            index % self.A_size
        ]  # make sure index is within then range
        A_img = Image.open(A_path).convert("RGB")
        if self.opt.serial_batches:  # make sure index is within then range
            index_B = index % self.B_size
        else:  # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        B_img = Image.open(B_path).convert("RGB")

        imgs = {
            "A_paths": A_path,
            "B_paths": B_path,
        }

        if self.opt.netG != "liss":
            B = self.transform_B(B_img)
            A = self.transform_A(A_img)
            imgs.update({"A": A, "B": B})
            return imgs

        A_d_img = Image.open(
            Path(A_path).parent / "depths" / (Path(A_path).stem + ".png")
        ).convert("L")
        B_d_img = Image.open(
            Path(B_path).parent / "depths" / (Path(B_path).stem + ".png")
        ).convert("L")

        im_dict_A = {"A_real": A_img}
        if "depth" in self.tasks.keys:
            im_dict_A["A_depth_target"] = A_d_img
        if "rotation" in self.tasks.keys:
            im_dict_A["A_rotation"] = A_img
        if "gray" in self.tasks.keys:
            im_dict_A["A_gray"] = A_img
        if "jigsaw" in self.tasks.keys:
            im_dict_A["A_jigsaw"] = A_img

        ims_A = self.transform_A(im_dict_A)
        if "rotation" in self.tasks.keys:
            ims_A["A_rotation_target"] = copy(ims_A["rotation_target"])
            del ims_A["rotation_target"]
        imgs.update(ims_A)

        im_dict_B = {"B_real": B_img}
        if "depth" in self.tasks.keys:
            im_dict_B["B_depth_target"] = B_d_img
        if "rotation" in self.tasks.keys:
            im_dict_B["B_rotation"] = B_img
        if "gray" in self.tasks.keys:
            im_dict_B["B_gray"] = B_img
        if "jigsaw" in self.tasks.keys:
            im_dict_B["B_jigsaw"] = B_img

        ims_B = self.transform_B(im_dict_B)
        if "rotation" in self.tasks.keys:
            ims_B["B_rotation_target"] = copy(ims_B["rotation_target"])
            del ims_B["rotation_target"]
        imgs.update(ims_B)

        return imgs

    def __len__(self):
        """Return the total number of images in the dataset.
        As we have two datasets with potentially different number of images,
        we take a maximum of
        """

        return max(self.A_size, self.B_size)
