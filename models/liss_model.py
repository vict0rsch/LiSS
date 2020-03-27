import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from util.util import angles_to_tensors
import torch.nn as nn
from .task import AuxiliaryTasks
import functools
import torch_optimizer as optim

DELIMITER = "."


def rgetattr(obj, path: str, *default):
    """
    Recursive getattr
    :param obj: Object
    :param path: 'attr1.attr2.etc'
    :param default: Optional default value, at any point in the path
    :return: obj.attr1.attr2.etc
    """

    attrs = path.split(DELIMITER)
    try:
        return functools.reduce(getattr, attrs, obj)
    except AttributeError:
        if default:
            return default[0]
        raise


class LiSSModel(BaseModel):
    """
    This class implements the LissModel model, for learning image-to-image
    translation without paired data, using self-supervised auxiliary tasks.
    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').
    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.
        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this
                flag to add training-specific or test-specific options.
        Returns:
            the modified parser.
        For CycleGAN, in addition to GAN losses, we introduce lambda_A, lambda_B,
        and lambda_I for the following losses.
        A (source domain), B (target domain).
        Generators: G_A: A -> B; G_B: B -> A.
        Discriminators: D_A: G_A(A) vs. B; D_B: G_B(B) vs. A.
        Forward cycle loss:  lambda_A * ||G_B(G_A(A)) - A|| (Eqn. (2) in the paper)
        Backward cycle loss: lambda_B * ||G_A(G_B(B)) - B|| (Eqn. (2) in the paper)
        Identity loss (optional):
        lambda_I * (||G_A(B) - B|| * lambda_B + ||G_B(A) - A|| * lambda_A)
        (Sec 5.2 "Photo generation from paintings" in the paper)
        Dropout is not used in the original CycleGAN paper.
        """
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument(
                "--lambda_CA",
                type=float,
                default=10.0,
                help="weight for cycle loss (A -> B -> A)",
            )
            parser.add_argument(
                "--lambda_CB",
                type=float,
                default=10.0,
                help="weight for cycle loss (B -> A -> B)",
            )
            parser.add_argument(
                "--lambda_DA",
                type=float,
                default=1.0,
                help="weight for Discriminator loss (A -> B -> A)",
            )
            parser.add_argument(
                "--lambda_DB",
                type=float,
                default=1.0,
                help="weight for Discriminator loss (B -> A -> B)",
            )
            parser.add_argument(
                "--lambda_R", type=float, default=1.0, help="weight for rotation"
            )
            parser.add_argument(
                "--lambda_J", type=float, default=1.0, help="weight for jigsaw"
            )
            parser.add_argument(
                "--lambda_D", type=float, default=1.0, help="weight for depth"
            )
            parser.add_argument(
                "--lambda_G", type=float, default=1.0, help="weight for gray"
            )
            parser.add_argument(
                "--lambda_DR",
                type=float,
                default=1.0,
                help="weight for rotation loss in discriminator (see SSGAN minimax)",
            )
            parser.add_argument(
                "--lambda_distillation",
                type=float,
                default=5.0,
                help="weight for distillation loss (when repr_mode == 'distillation')",
            )
            parser.add_argument(
                "--lambda_I",
                type=float,
                default=0.5,
                help="use identity mapping. Setting lambda_I other than 0 has an\
                    effect of scaling the weight of the identity mapping loss. For \
                    example, if the weight of the identity loss should be 10 times \
                    smaller than the weight of the reconstruction loss, please set \
                    lambda_I = 0.1",
            )
            parser.add_argument(
                "--task_schedule", type=str, default="parallel", help="Tasks schedule"
            )
            # sequential       :  <rotation> then <depth> then <translation>
            #                     without the possibility to come back
            #
            # parallel         :  always <depth, rotation, translation>
            #
            # additional       :  <rotation> then <depth, rotation> then
            #                     <depth, rotation, translation>
            #
            # liss             :  sequential with distillation
            #
            # representational :  <rotation, depth, identity> then <translation>
            parser.add_argument(
                "--rotation_acc_threshold",
                type=float,
                default=0.2,
                help="minimal rotation classification loss to switch task",
            )
            parser.add_argument(
                "--jigsaw_acc_threshold",
                type=float,
                default=0.2,
                help="minimal jigsaw classification loss to switch task",
            )
            parser.add_argument(
                "--depth_loss_threshold",
                type=float,
                default=0.5,
                help="minimal depth estimation loss to switch task",
            )
            parser.add_argument(
                "--gray_loss_threshold",
                type=float,
                default=0.5,
                help="minimal gray loss to switch task",
            )
            parser.add_argument(
                "--i_loss_threshold",
                type=float,
                default=0.5,
                help="minimal identity loss to switch task (representational only)",
            )
            parser.add_argument(
                "--lr_rotation",
                type=float,
                default=0,
                help="minimal identity loss to switch task (representational only)",
            )
            parser.add_argument(
                "--lr_depth",
                type=float,
                default=0,
                help="minimal identity loss to switch task (representational only)",
            )
            parser.add_argument("--lr_gray", type=float, default=0)
            parser.add_argument("--lr_jigsaw", type=float, default=0)
            parser.add_argument(
                "--encoder_merge_ratio",
                type=float,
                default=1.0,
                help="Exp. moving average coefficient: ref = a * new + (1 - a) * old",
            )
            parser.add_argument(
                "--auxiliary_tasks", type=str, default="rotation, gray, depth, jigsaw"
            )
            parser.add_argument(
                "--D_rotation",
                action="store_true",
                default=False,
                help="use rotation self-supervision in discriminators when translating",
            )
            parser.add_argument(
                "--repr_mode",
                type=str,
                default="freeze",
                help="freeze | flow | distillation: When switching from representation "
                + "to traduction: either freeze the encoder or let gradients flow. "
                + "Set continual for flow + distillation loss",
            )

        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.
        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be
                a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        self.tasks = AuxiliaryTasks([t.strip() for t in opt.auxiliary_tasks.split(",")])
        # specify the training losses you want to print out.
        # The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = [
            "D_A",
            "G_A",
            "cycle_A",
            "A_idt",
            "D_B",
            "G_B",
            "cycle_B",
            "B_idt",
            "G_D_rotation",
            "G",
        ]
        if opt.task_schedule == "liss" or opt.repr_mode == "distillation":
            self.loss_names += ["G_distillation_A", "G_distillation_B"]

        for t in self.tasks:
            self.loss_names += t.loss_names

        # combine visualizations for A and B
        self.visual_names = set(["A_real", "B_real"])
        # specify the models you want to save to the disk. The training/test scripts
        # will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ["G_A", "G_B", "D_A", "D_B"]
        else:  # during test time, only load Gs
            self.model_names = ["G_A", "G_B"]

        self.exp = None

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_A = networks.define_G(
            opt.input_nc,
            opt.output_nc,
            opt.ngf,
            opt.netG,
            opt.norm,
            not opt.no_dropout,
            opt.init_type,
            opt.init_gain,
            self.gpu_ids,
        )
        self.netG_B = networks.define_G(
            opt.output_nc,
            opt.input_nc,
            opt.ngf,
            opt.netG,
            opt.norm,
            not opt.no_dropout,
            opt.init_type,
            opt.init_gain,
            self.gpu_ids,
        )
        self.is_dp = isinstance(self.netG_A, nn.DataParallel)
        self.ref_encoder_A = self.ref_encoder_B = None

        if self.isTrain:  # define discriminators
            self.netD_A = networks.define_D(
                opt.output_nc,
                opt.ndf,
                opt.netD,
                opt.n_layers_D,
                opt.norm,
                opt.init_type,
                opt.init_gain,
                self.gpu_ids,
            )
            self.netD_B = networks.define_D(
                opt.input_nc,
                opt.ndf,
                opt.netD,
                opt.n_layers_D,
                opt.norm,
                opt.init_type,
                opt.init_gain,
                self.gpu_ids,
            )
            for t in self.tasks:
                if t.needs_D:
                    setattr(
                        self,
                        f"netD_A_{t.key}",
                        networks.define_D(
                            opt.output_nc,
                            opt.ndf,
                            opt.netD,
                            opt.n_layers_D,
                            opt.norm,
                            opt.init_type,
                            opt.init_gain,
                            self.gpu_ids,
                        ),
                    )
                    setattr(
                        self,
                        f"netD_B_{t.key}",
                        networks.define_D(
                            opt.output_nc,
                            opt.ndf,
                            opt.netD,
                            opt.n_layers_D,
                            opt.norm,
                            opt.init_type,
                            opt.init_gain,
                            self.gpu_ids,
                        ),
                    )

        if self.isTrain:
            if opt.lambda_I > 0.0:
                # only works when input and output images have the same number of
                # channels
                assert opt.input_nc == opt.output_nc
            self.fake_A_pool = ImagePool(opt.pool_size)
            # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)
            # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionGray = torch.nn.L1Loss()
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            self.rotationCriterion = torch.nn.CrossEntropyLoss()
            self.jigsawCriterion = torch.nn.CrossEntropyLoss()
            self.depthCriterion = torch.nn.L1Loss()
            self.distillationCriterion = torch.nn.L1Loss()

            if self.is_dp:
                params = {
                    "params": itertools.chain(
                        self.netG_A.module.encoder.parameters(),
                        self.netG_A.module.decoder.parameters(),
                        self.netG_B.module.encoder.parameters(),
                        self.netG_B.module.decoder.parameters(),
                    ),
                    "group_name": "default",
                }
                all_G_params = [params]
                for t in self.tasks:
                    tp = itertools.chain(
                        getattr(self.netG_A.module, t.module_name).parameters(),
                        getattr(self.netG_B.module, t.module_name).parameters(),
                    )
                    if t.needs_lr:
                        if getattr(opt, "lr_" + t.key) > 0:
                            lr = getattr(opt, "lr_" + t.key)
                        else:
                            lr = opt.lr
                        all_G_params += [{"group_name": t.key, "params": tp, "lr": lr}]
                    else:
                        all_G_params += [{"group_name": t.key, "params": tp}]
            else:
                params = {
                    "params": itertools.chain(
                        self.netG_A.encoder.parameters(),
                        self.netG_A.decoder.parameters(),
                        self.netG_B.encoder.parameters(),
                        self.netG_B.decoder.parameters(),
                    ),
                    "group_name": "default",
                }
                all_G_params = [params]
                for t in self.tasks:
                    tp = itertools.chain(
                        getattr(self.netG_A, t.module_name).parameters(),
                        getattr(self.netG_B, t.module_name).parameters(),
                    )
                    if t.needs_lr:
                        if getattr(opt, "lr_" + t.key) > 0:
                            lr = getattr(opt, "lr_" + t.key)
                        else:
                            lr = opt.lr
                        all_G_params += [{"group_name": t.key, "params": tp, "lr": lr}]
                    else:
                        all_G_params += [{"group_name": t.key, "params": tp}]

            self.optimizer_G = optim.RAdam(
                all_G_params, lr=opt.lr, betas=(opt.beta1, 0.999)
            )
            all_D_params = []
            for d in dir(self):
                if d.startswith("netD_"):
                    attr = self.get(d)
                    assert isinstance(attr, nn.Module)
                    all_D_params.append(attr.parameters())
            self.optimizer_D = optim.RAdam(
                itertools.chain(*all_D_params), lr=opt.lr, betas=(opt.beta1, 0.999)
            )
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
        self.init_schedule()

    def get_state_dict(self):
        return {"net" + k: self.get("net" + k).state_dict() for k in self.model_names}

    def set_state_dict(self, d):
        for k, v in d.items():
            self.get(k).load_state_dict(v)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == "AtoB"
        self.A_real = input["A_real" if AtoB else "B_real"].to(self.device)
        self.image_paths_A = input["A_paths" if AtoB else "B_paths"]
        self.B_real = input["B_real" if AtoB else "A_real"].to(self.device)
        self.image_paths_B = input["B_paths" if AtoB else "A_paths"]

        self.current_input = input

        for t in self.tasks:
            # ------------------------
            # -----  Input data  -----
            # ------------------------
            if ("A_" + t.input_key) in input and ("B_" + t.input_key) in input:
                inputs = [
                    input["A_" + t.input_key if AtoB else "B_" + t.input_key],
                    input["B_" + t.input_key if AtoB else "A_" + t.input_key],
                ]
                for i, inp in enumerate(inputs):
                    inputs[i] = inp.to(self.device).float()

                if t.key == "rotation":
                    for i, inp in enumerate(inputs):
                        inputs[i] = inp.view(
                            -1, 3, self.A_real.shape[-2], self.A_real.shape[-1]
                        )
                setattr(self, "A_" + t.key, inputs[0])
                setattr(self, "B_" + t.key, inputs[1])
            # -------------------------
            # -----  Target data  -----
            # -------------------------
            if ("A_" + t.target_key) in input:
                targets = [
                    input["A_" + t.target_key if AtoB else "B_" + t.target_key],
                    input["B_" + t.target_key if AtoB else "A_" + t.target_key],
                ]
                for i, tar in enumerate(targets):
                    targets[i] = tar.to(self.device)

                if t.key == "rotation":
                    for i, target in enumerate(targets):
                        targets[i] = target.view(-1)

                setattr(self, "A_" + t.target_key, targets[0])
                setattr(self, "B_" + t.target_key, targets[1])

    def forward(self, ignore=set(), force=set()):
        """Run forward pass; called by both functions
        <optimize_parameters> and <test>."""

        assert not (ignore & force)
        should = {"identity": False, "translation": False}
        should.update({t.key: False for t in self.tasks})

        for k in should:
            if k in force:
                should[k] = True
            elif k in ignore:
                should[k] = False
            else:
                should[k] = self.should_compute(k)

        # -------------------------------
        # -----  DataParallel Mode  -----
        # -------------------------------
        if self.is_dp:
            # --------------------
            # -----  Encode  -----
            # --------------------
            self.A_z = self.netG_A.module.encoder(self.A_real)
            self.B_z = self.netG_B.module.encoder(self.B_real)
            if (
                self.opt.repr_mode == "distillation" and self.repr_is_frozen
            ) or self.opt.task_schedule == "liss":
                if self.ref_encoder_A is not None:
                    self.A_z_ref = self.ref_encoder_A(self.A_real)
                    self.B_z_ref = self.ref_encoder_B(self.B_real)

            if "identity" in should and should["identity"]:
                self.A_idt = self.netG_A.module.decoder(self.B_z)
                self.B_idt = self.netG_B.module.decoder(self.A_z)

            if "translation" in should and should["translation"]:
                self.B_fake = self.netG_A.module.decoder(self.A_z)  # G_A(A)
                self.A_rec = self.netG_B(self.B_fake)  # G_B(G_A(A))
                self.A_fake = self.netG_B.module.decoder(self.B_z)  # G_B(B)
                self.B_rec = self.netG_A(self.A_fake)  # G_A(G_B(B))

            for t in self.tasks:
                if t.key not in should or not should[t.key]:
                    continue

                if t.needs_z:
                    z_data = {
                        domain: self.get(f"{domain}_{t.key}") for domain in ["A", "B"]
                    }
                    for domain in ["A", "B"]:
                        encoder = rgetattr(self, f"netG_{domain}.module.encoder")
                        z_data[domain] = encoder(z_data[domain])
                        setattr(self, f"{domain}_z_{t.key}", z_data)
                else:
                    z_data = {"A": self.A_z, "B": self.B_z}

                for domain in ["A", "B"]:
                    model = rgetattr(self, f"netG_{domain}.module.{t.module_name}")
                    setattr(self, f"{domain}_{t.key}_pred", model(z_data[domain]))

        # ---------------------------
        # -----  Other Devices  -----
        # ---------------------------
        else:
            # --------------------
            # -----  Encode  -----
            # --------------------
            self.A_z = self.netG_A.encoder(self.A_real)
            self.B_z = self.netG_B.encoder(self.B_real)
            if (
                self.opt.repr_mode == "distillation" and self.repr_is_frozen
            ) or self.opt.task_schedule == "liss":
                if self.ref_encoder_A is not None:
                    self.A_z_ref = self.ref_encoder_A(self.A_real)
                    self.B_z_ref = self.ref_encoder_B(self.B_real)

            if "identity" in should and should["identity"]:
                self.A_idt = self.netG_A.decoder(self.B_z)
                self.B_idt = self.netG_B.decoder(self.A_z)

            if "translation" in should and should["translation"]:
                self.B_fake = self.netG_A.decoder(self.A_z)  # G_A(A)
                self.A_rec = self.netG_B(self.B_fake)  # G_B(G_A(A))
                self.A_fake = self.netG_B.decoder(self.B_z)  # G_B(B)
                self.B_rec = self.netG_A(self.A_fake)  # G_A(G_B(B))

            for t in self.tasks:
                if t.key not in should or not should[t.key]:
                    continue

                if t.needs_z:
                    z_data = {
                        domain: self.get(f"{domain}_{t.key}") for domain in ["A", "B"]
                    }
                    for domain in ["A", "B"]:
                        encoder = rgetattr(self, f"netG_{domain}.encoder")
                        z_data[domain] = encoder(z_data[domain])
                        setattr(self, f"{domain}_z_{t.key}", z_data)
                else:
                    z_data = {"A": self.A_z, "B": self.B_z}

                for domain in ["A", "B"]:
                    model = rgetattr(self, f"netG_{domain}.{t.module_name}")
                    setattr(self, f"{domain}_{t.key}_pred", model(z_data[domain]))

    def loss_D_basic(self, netD, real, fake, domain="A"):
        """Calculate GAN loss for the discriminator
        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator
        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real, _ = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake, _ = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5

        if self.opt.D_rotation:
            rotation_input = self.get(f"{domain}_rotation")
            rotation_target = self.get(f"{domain}_rotation_target")
            _, rotation_pred = netD(rotation=rotation_input)
            loss_D += self.rotationCriterion(rotation_pred, rotation_target)

        # loss_D.backward()
        return loss_D

    def backward_D(self):

        if self.should_compute("translation"):
            if (
                self.opt.task_schedule != "representational-traduction"
                or self.repr_is_frozen
            ):
                self.set_requires_grad([self.netD_A, self.netD_B], True)
                self.backward_D_A()
                self.backward_D_B()

        for t in self.tasks:
            if not self.should_compute(t.key) or not t.needs_D:
                continue
            for domain in ["A", "B"]:
                netD = self.get(f"netD_{domain}_{t.key}")
                self.set_requires_grad([netD], True)
                real = self.get(f"{domain}_{t.target_key}")
                fake = self.get(f"{domain}_{t.key}_pred")
                loss = self.loss_D_basic(netD, real, fake, domain)
                loss.backward()
                setattr(self, f"loss_D_{domain}_{t.key}", loss)

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        B_fake = self.fake_B_pool.query(self.B_fake)
        self.loss_D_A = self.loss_D_basic(self.netD_A, self.B_real, B_fake, "A")
        self.loss_D_A_basic = self.loss_D_A.clone()
        if self.opt.D_rotation:
            mixed = []
            mixed_target = []
            for b_idx in range(len(self.A_real)):
                fake = self.A_fake[b_idx].unsqueeze(0)
                real = self.A_rotation[b_idx * 4 : (b_idx + 1) * 4]
                rotation_target = self.A_rotation_target[b_idx * 4 : (b_idx + 1) * 4]
                fake_idx = int(torch.randint(0, 5, (1,)))
                mixed.append(torch.cat([real[:fake_idx], fake, real[fake_idx:]], dim=0))
                mixed_target.append(
                    torch.cat(
                        [
                            rotation_target[:fake_idx],
                            torch.tensor([4], device=self.device),
                            rotation_target[fake_idx:],
                        ],
                        dim=0,
                    )
                )
            mixed = torch.cat(mixed, dim=0).detach()
            mixed_target = torch.cat(mixed_target, dim=0)
            self.loss_D_A_rotation = self.rotationCriterion(
                self.netD_A(rotation=mixed)[1], mixed_target.to(self.device)
            )
            self.loss_D_A += self.loss_D_A_rotation

        self.loss_D_A.backward()

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        A_fake = self.fake_A_pool.query(self.A_fake)
        self.loss_D_B = self.loss_D_basic(self.netD_B, self.A_real, A_fake, "B")
        self.loss_D_B_basic = self.loss_D_B.clone()
        if self.opt.D_rotation:
            mixed = []
            mixed_target = []
            for b_idx in range(len(self.B_real)):
                fake = self.B_fake[b_idx].unsqueeze(0)
                real = self.B_rotation[b_idx * 4 : (b_idx + 1) * 4]
                rotation_target = self.B_rotation_target[b_idx * 4 : (b_idx + 1) * 4]
                fake_idx = int(torch.randint(0, 5, (1,)))
                mixed.append(torch.cat([real[:fake_idx], fake, real[fake_idx:]], dim=0))
                mixed_target.append(
                    torch.cat(
                        [
                            rotation_target[:fake_idx],
                            torch.tensor([4], device=self.device),
                            rotation_target[fake_idx:],
                        ],
                        dim=0,
                    )
                )
            mixed = torch.cat(mixed, dim=0).detach()
            mixed_target = torch.cat(mixed_target, dim=0)

            self.loss_D_B_rotation = self.rotationCriterion(
                self.netD_B(rotation=mixed)[1], mixed_target.to(self.device)
            )
            self.loss_D_B += self.loss_D_B_rotation

        self.loss_D_B.backward()

    def should_compute(self, arg):
        key = f"_should_compute_{arg}"
        if not hasattr(self, key):
            raise ValueError(f"Unknown arg {arg}")
        return self.get(key)

    def backward_G(self, losses_only=False, ignore=set(), force=set()):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.opt.lambda_I
        lambda_CA = self.opt.lambda_CA
        lambda_DA = self.opt.lambda_DA
        lambda_CB = self.opt.lambda_CB
        lambda_DB = self.opt.lambda_DB
        lambda_D = self.opt.lambda_D
        lambda_R = self.opt.lambda_R
        lambda_G = self.opt.lambda_G
        lambda_J = self.opt.lambda_J
        lambda_distillation = self.opt.lambda_distillation

        lambda_total = 0

        assert not (ignore & force)
        should = {"identity": False, "translation": False}
        should.update({t.key: False for t in self.tasks})

        for k in should:
            if k in force:
                should[k] = True
            elif k in ignore:
                should[k] = False
            else:
                should[k] = self.should_compute(k)

        self.loss_G = 0
        if "depth" in should and should["depth"]:
            # print("depth loss")
            device = self.A_depth_pred.device
            self.loss_G_A_depth = self.depthCriterion(
                self.A_depth_pred, self.A_depth_target.to(device).float()
            )
            self.loss_G_B_depth = self.depthCriterion(
                self.B_depth_pred, self.B_depth_target.to(device).float()
            )
            self.loss_G += lambda_D * (self.loss_G_B_depth + self.loss_G_A_depth)
            lambda_total += 2 * lambda_D

        if "rotation" in should and should["rotation"]:
            # print("rotation loss")
            device = self.A_rotation_pred.device
            self.loss_G_A_rotation = self.rotationCriterion(
                self.A_rotation_pred,
                angles_to_tensors(self.A_rotation_target, one_hot=False).to(device),
            )
            self.loss_G_B_rotation = self.rotationCriterion(
                self.B_rotation_pred,
                angles_to_tensors(self.B_rotation_target, one_hot=False).to(device),
            )
            self.loss_G += lambda_R * (self.loss_G_B_rotation + self.loss_G_A_rotation)
            lambda_total += 2 * lambda_R

        if "jigsaw" in should and should["jigsaw"]:
            # print("jigsaw loss")
            device = self.A_jigsaw_pred.device
            self.loss_G_A_jigsaw = self.jigsawCriterion(
                self.A_jigsaw_pred, self.A_jigsaw_target.to(device)
            )
            self.loss_G_B_jigsaw = self.jigsawCriterion(
                self.B_jigsaw_pred, self.B_jigsaw_target.to(device)
            )
            self.loss_G += lambda_J * (self.loss_G_B_jigsaw + self.loss_G_A_jigsaw)
            lambda_total += 2 * lambda_J

        if "identity" in should and should["identity"]:
            # print("identity loss")
            # Identity loss
            assert lambda_idt > 0
            # G_A should be identity if B_real is fed: ||G_A(B) - B||
            self.loss_G_A_idt = (
                self.criterionIdt(self.A_idt, self.B_real) * lambda_CB * lambda_idt
            )
            # G_B should be identity if A_real is fed: ||G_B(A) - A||
            self.loss_G_B_idt = (
                self.criterionIdt(self.B_idt, self.A_real) * lambda_CA * lambda_idt
            )

            self.loss_G += self.loss_G_A_idt + self.loss_G_B_idt
            lambda_total += lambda_CA * lambda_idt + lambda_CB * lambda_idt

        if "gray" in should and should["gray"]:
            self.loss_G_A_gray = 0.1 * self.criterionGAN(
                self.netD_A_gray(self.A_gray_pred)[0], True
            ) + 0.9 * self.criterionGray(self.A_gray_pred, self.A_real)
            self.loss_G_B_gray = 0.1 * self.criterionGAN(
                self.netD_B_gray(self.B_gray_pred)[0], True
            ) + 0.9 * self.criterionGray(self.B_gray_pred, self.B_real)

            self.loss_G += (self.loss_G_A_gray + self.loss_G_B_gray) * lambda_G
            lambda_total += 2 * lambda_G

        if "translation" in should and should["translation"]:
            # print("translation loss")
            # GAN loss D_A(G_A(A))
            self.loss_G_A = (
                self.criterionGAN(self.netD_A(self.B_fake)[0], True) * lambda_DA
            )
            # GAN loss D_B(G_B(B))
            self.loss_G_B = (
                self.criterionGAN(self.netD_B(self.A_fake)[0], True) * lambda_DB
            )
            # Forward cycle loss || G_B(G_A(A)) - A||
            self.loss_cycle_A = self.criterionCycle(self.A_rec, self.A_real) * lambda_CA
            # Backward cycle loss || G_A(G_B(B)) - B||
            self.loss_cycle_B = self.criterionCycle(self.B_rec, self.B_real) * lambda_CB
            # combined loss and calculate gradients
            self.loss_G_D_rotation = 0
            if self.opt.D_rotation:
                for domain in "AB":
                    angles = [torch.randperm(4) for _ in range(len(self.A_real))]
                    fake_rotations_target = torch.cat(
                        [a.unsqueeze(0) for a in angles], dim=0
                    ).view(-1)
                    fake_rotations = []
                    for i, perm in enumerate(angles):
                        rotated = self.get(f"{domain}_fake")[i].unsqueeze(0)
                        fake_rotations.append(
                            torch.cat(
                                [
                                    torch.rot90(rotated, k=p, dims=[-2, -1])
                                    for p in perm
                                ],
                                dim=0,
                            )
                        )
                    fake_rotations = torch.cat(fake_rotations, dim=0)
                    _, fake_rotations_pred = self.get(f"netD_{domain}")(
                        rotation=fake_rotations
                    )
                    _loss_fake = self.rotationCriterion(
                        fake_rotations_pred, fake_rotations_target.to(self.device)
                    )
                    _, real_rotations_pred = self.get(f"netD_{domain}")(
                        rotation=self.get(f"{domain}_rotation")
                    )
                    _loss_real = self.rotationCriterion(
                        real_rotations_pred, self.get(f"{domain}_rotation_target")
                    )
                    _loss = torch.abs(_loss_fake - _loss_real)
                    self.set(f"loss_G_{domain}_D_rotation", _loss)
                self.loss_G_D_rotation = self.opt.lambda_DR * (
                    self.loss_G_A_D_rotation + self.loss_G_B_D_rotation
                )
                lambda_total += 2 * self.opt.lambda_DR
            self.loss_G += (
                self.loss_G_A
                + self.loss_G_B
                + self.loss_cycle_A
                + self.loss_cycle_B
                + self.loss_G_D_rotation
            )

        # --------------------------
        # -----  Distillation  -----
        # --------------------------
        if (
            self.opt.repr_mode == "distillation" and self.repr_is_frozen
        ) or self.opt.task_schedule == "liss":
            if self.ref_encoder_A is not None:
                self.loss_G_distillation_A = (
                    self.distillationCriterion(self.A_z, self.A_z_ref)
                    * lambda_distillation
                )
                self.loss_G_distillation_B = (
                    self.distillationCriterion(self.B_z, self.B_z_ref)
                    * lambda_distillation
                )
                lambda_total += 2 * lambda_distillation
                self.loss_G += self.loss_G_distillation_A + self.loss_G_distillation_B
            else:
                self.loss_G_distillation_A = self.loss_G_distillation_B = 0

        lambda_total += lambda_CA + lambda_CB

        scale = False
        if scale:
            self.loss_G /= lambda_total
            self.exp.log_parameter("scale_loss_with_lambda_total", True)
        else:
            self.exp.log_parameter("scale_loss_with_lambda_total", False)

        if not losses_only:
            self.exp.log_metric("lambda_total", lambda_total)
            self.loss_G.backward()

    def sequential_schedule(self, metrics):
        for t in self.tasks:
            threshold = getattr(self.opt, t.threshold_key)
            metric_A = metrics[f"test_G_A_{t.key}_{t.threshold_type}"]
            metric_B = metrics[f"test_G_B_{t.key}_{t.threshold_type}"]
            s = f"{t.key}_{t.threshold_type} : "
            s += f"{metric_A} & {metric_B} vs {threshold}\n"
            if t.threshold_type == "acc":
                condition = metric_A > threshold and metric_B > threshold
            else:
                condition = metric_A < threshold and metric_B < threshold

            condition = condition and self.should_compute(t.key)

            if condition:
                next_key = self.tasks.task_after(t.key)
                if next_key is not None:
                    next_keys = [next_key]
                else:
                    next_keys = ["identity", "translation"]

                for i, nk in enumerate(next_keys):
                    setattr(self, f"_should_compute_{t.key}", False)
                    setattr(self, f"_should_compute_{nk}", True)
                    print(f"\n\n>> Stop {t.key} ; Start {next_keys} <<\n")
                    if self.exp:
                        if i == 0:
                            if f"schedule_stop_{t.key}" not in self.exp.params:
                                self.exp.log_parameter(
                                    f"schedule_stop_{t.key}", self.total_iters
                                )
                                self.exp.log_text(
                                    f"schedule_stop_{t.key}: {self.total_iters}"
                                )
                        if f"schedule_start_{nk}" not in self.exp.params:
                            self.exp.log_parameter(
                                f"schedule_start_{nk}", self.total_iters
                            )
                            self.exp.log_text(
                                f"schedule_start_{nk}: {self.total_iters}"
                            )
                return
            else:
                print("No schedule update: " + s)
                if self.exp:
                    self.exp.log_text(
                        "No schedule update: " + s + f" ({self.total_iters})"
                    )

    def liss_schedule(self, metrics):
        s = ""
        for t in self.tasks:
            threshold = getattr(self.opt, t.threshold_key)
            metric_A = metrics[f"test_G_A_{t.key}_{t.threshold_type}"]
            metric_B = metrics[f"test_G_B_{t.key}_{t.threshold_type}"]
            s += f"{t.key}_{t.threshold_type} : "
            s += f"{metric_A} & {metric_B} vs {threshold}\n"
            if t.threshold_type == "acc":
                condition = metric_A > threshold and metric_B > threshold
            else:
                condition = metric_A < threshold and metric_B < threshold

            condition = condition and self.should_compute(t.key)

            if condition:
                next_key = self.tasks.task_after(t.key)
                if next_key is not None:
                    next_keys = [next_key]
                else:
                    next_keys = ["identity", "translation"]

                for i, nk in enumerate(next_keys):
                    setattr(self, f"_should_compute_{t.key}", False)
                    setattr(self, f"_should_compute_{nk}", True)
                    print(f"\n\n>> Stop {t.key} ; Start {next_keys} <<\n")
                    if self.exp:
                        if i == 0:
                            if f"schedule_stop_{t.key}" not in self.exp.params:
                                self.exp.log_parameter(
                                    f"schedule_stop_{t.key}", self.total_iters
                                )
                                self.exp.log_text(
                                    f"schedule_stop_{t.key}: {self.total_iters}"
                                )
                        if f"schedule_start_{nk}" not in self.exp.params:
                            self.exp.log_parameter(
                                f"schedule_start_{nk}", self.total_iters
                            )
                            self.exp.log_text(
                                f"schedule_start_{nk}: {self.total_iters}"
                            )
                self.update_ref_encoder()
                return
        if not self.should_compute("translation"):
            print("No schedule update: " + s)
            if self.exp:
                self.exp.log_text("No schedule update: " + s + f" ({self.total_iters})")

    def additional_schedule(self, metrics):
        for t in self.tasks:
            threshold = getattr(self.opt, t.threshold_key)
            metric_A = metrics[f"test_G_A_{t.key}_{t.threshold_type}"]
            metric_B = metrics[f"test_G_B_{t.key}_{t.threshold_type}"]
            s = f"{t.key}_{t.threshold_type} : "
            s += f"{metric_A} & {metric_B} vs {threshold}\n"
            if t.threshold_type == "acc":
                condition = metric_A > threshold and metric_B > threshold
            else:
                condition = metric_A < threshold and metric_B < threshold

            if condition:
                next_key = self.tasks.task_after(t.key)
                if next_key is not None:
                    next_keys = [next_key]
                else:
                    next_keys = ["identity", "translation"]
                for i, nk in enumerate(next_keys):
                    setattr(self, f"_should_compute_{nk}", False)
                    if i == 0:
                        print(f"\n\n>> Start {next_keys} <<\n")
                    if self.exp:
                        if f"schedule_start_{nk}" not in self.exp.params:
                            self.exp.log_parameter(
                                f"schedule_start_{nk}", self.total_iters
                            )
                            self.exp.log_text(
                                f"schedule_start_{nk}: {self.total_iters}"
                            )
            else:
                print("No update for " + s)
                if self.exp:
                    self.exp.log_text("No update for " + s + f" ({self.total_iters})")

    def representational_traduction_schedule(self, metrics):
        task_conditions = True
        s = ""
        for t in self.tasks:
            threshold = getattr(self.opt, t.threshold_key)
            metric_A = metrics[f"test_G_A_{t.key}_{t.threshold_type}"]
            metric_B = metrics[f"test_G_B_{t.key}_{t.threshold_type}"]
            s += f"{t.key}_{t.threshold_type} : "
            s += f"{metric_A} & {metric_B} vs {threshold}\n"
            if t.threshold_type == "acc":
                task_condition = metric_A > threshold and metric_B > threshold
            else:
                task_condition = metric_A < threshold and metric_B < threshold
            task_conditions = task_conditions and task_condition

        # i = self.opt.i_loss_threshold

        if task_conditions:
            print("\n\n>> Start translation <<\n")
            self._should_compute_translation = True
            self._should_compute_identity = True
            for t in self.tasks:
                setattr(self, f"_should_compute_{t.key}", False)

            if not self.repr_is_frozen:
                print(">>> Freezing")
                if self.exp:
                    if "schedule_start_translation" not in self.exp.params:
                        self.exp.log_parameter(
                            "schedule_start_translation", self.total_iters
                        )
                        self.exp.log_parameter(
                            "schedule_stop_representation", self.total_iters
                        )
                        self.exp.log_text(
                            f"schedule_start_translation {self.total_iters}"
                        )
                        self.exp.log_text(
                            f"schedule_stop_representation {self.total_iters}"
                        )
                if self.is_dp:
                    freeze = (
                        [self.netG_A.module.encoder, self.netG_B.module.encoder]
                        if self.opt.repr_mode == "freeze"
                        else []
                    )
                    freeze += [
                        rgetattr(self, f"netG_{d}.module.{t.module_name}")
                        for d in ["A", "B"]
                        for t in self.tasks
                    ]
                    unfreeze = [self.netG_A.module.decoder, self.netG_B.module.decoder]
                else:
                    freeze = (
                        [self.netG_A.encoder, self.netG_B.encoder]
                        if self.opt.repr_mode == "freeze"
                        else []
                    )
                    freeze += [
                        rgetattr(self, f"netG_{d}.{t.module_name}")
                        for d in ["A", "B"]
                        for t in self.tasks
                    ]
                    unfreeze = [self.netG_A.decoder, self.netG_B.decoder]
                self.set_requires_grad(freeze, requires_grad=False)
                self.set_requires_grad(unfreeze, requires_grad=True)

                if self.opt.repr_mode == "distillation":
                    print(">>> Distillation")
                    self.update_ref_encoder()

                self.repr_is_frozen = True
        else:
            print("No schedule update:\n" + s)
            if self.exp:
                self.exp.log_text("No schedule update:\n" + s + f"({self.total_iters})")

    def representational_schedule(self, metrics):

        task_conditions = True
        s = ""
        for t in self.tasks:
            threshold = getattr(self.opt, t.threshold_key)
            metric_A = metrics[f"test_G_A_{t.key}_{t.threshold_type}"]
            metric_B = metrics[f"test_G_B_{t.key}_{t.threshold_type}"]
            s += f"{t.key}_{t.threshold_type} : "
            s += f"{metric_A} & {metric_B} vs {threshold}\n"
            if t.threshold_type == "acc":
                task_condition = metric_A > threshold and metric_B > threshold
            else:
                task_condition = metric_A < threshold and metric_B < threshold
            task_conditions = task_conditions and task_condition

        # i = self.opt.i_loss_threshold

        if task_conditions:
            print("\n\n>> Start translation <<\n")
            if self.exp and "schedule_start_translation" not in self.exp.params:
                self.exp.log_parameter("schedule_start_translation", self.total_iters)
                self.exp.log_parameter("schedule_stop_representation", self.total_iters)
                self.exp.log_text(f"schedule_start_translation: {self.total_iters}")
                self.exp.log_text(f"schedule_stop_representation: {self.total_iters}")
            self._should_compute_translation = True
            self._should_compute_identity = True
            for t in self.tasks:
                setattr(self, f"_should_compute_{t.key}", False)

            if not self.repr_is_frozen:
                if self.is_dp:
                    freeze = (
                        [self.netG_A.module.encoder, self.netG_B.module.encoder]
                        if self.opt.repr_mode == "freeze"
                        else []
                    )
                    freeze += [
                        rgetattr(self, f"netG_{d}.module.{t.module_name}")
                        for d in ["A", "B"]
                        for t in self.tasks
                    ]
                    unfreeze = [self.netG_A.module.decoder, self.netG_B.module.decoder]
                else:
                    freeze = (
                        [self.netG_A.encoder, self.netG_B.encoder]
                        if self.opt.repr_mode == "freeze"
                        else []
                    )
                    freeze += [
                        rgetattr(self, f"netG_{d}.{t.module_name}")
                        for d in ["A", "B"]
                        for t in self.tasks
                    ]
                    unfreeze = [self.netG_A.decoder, self.netG_B.decoder]
                self.set_requires_grad(freeze, requires_grad=False)
                self.set_requires_grad(unfreeze, requires_grad=True)

                if self.opt.repr_mode == "distillation":
                    print(">>> Distillation")
                    self.update_ref_encoder()
                self.repr_is_frozen = True
        else:
            print("No schedule update:\n" + s)
            if self.exp:
                self.exp.log_text(
                    "No schedule update:\n" + s + f" ({self.total_iters})"
                )

    def update_ref_encoder(self):
        alpha = self.opt.encoder_merge_ratio
        if self.ref_encoder_A is None:
            if self.is_dp:
                self.ref_encoder_A = self.netG_A.module.get_encoder()
                self.ref_encoder_A.load_state_dict(
                    self.netG_A.module.encoder.state_dict()
                )
                self.ref_encoder_A = nn.DataParallel(
                    self.ref_encoder_A.to(self.opt.gpu_ids[0]), self.opt.gpu_ids
                )
                self.ref_encoder_B = self.netG_B.module.get_encoder()
                self.ref_encoder_B.load_state_dict(
                    self.netG_B.module.encoder.state_dict()
                )
                self.ref_encoder_B = nn.DataParallel(
                    self.ref_encoder_B.to(self.opt.gpu_ids[0]), self.opt.gpu_ids
                )
            else:
                print("No update_ref_encoder without gpus")

        else:
            if self.is_dp:
                new_encoder_A = self.netG_A.module.get_encoder()
                new_encoder_A.load_state_dict(
                    {
                        k: alpha * v1 + (1 - alpha) * v2
                        for (k, v1), (_, v2) in zip(
                            self.netG_A.module.encoder.state_dict().items(),
                            self.ref_encoder_A.module.state_dict().items(),
                        )
                    }
                )
                new_encoder_A = nn.DataParallel(
                    new_encoder_A.to(self.opt.gpu_ids[0]), self.opt.gpu_ids
                )
                new_encoder_B = self.netG_B.module.get_encoder()
                new_encoder_B.load_state_dict(
                    {
                        k: alpha * v1 + (1 - alpha) * v2
                        for (k, v1), (_, v2) in zip(
                            self.netG_B.module.encoder.state_dict().items(),
                            self.ref_encoder_B.module.state_dict().items(),
                        )
                    }
                )
                new_encoder_B = nn.DataParallel(
                    new_encoder_B.to(self.opt.gpu_ids[0]), self.opt.gpu_ids
                )

                self.ref_encoder_A = new_encoder_A
                self.ref_encoder_B = new_encoder_B
            else:
                print("No update_ref_encoder without gpus")

        self.set_requires_grad([self.ref_encoder_A, self.ref_encoder_B], False)

    def parallel_schedule(self, metrics):
        return

    def init_schedule(self):
        if self.opt.task_schedule == "parallel":
            for t in self.tasks:
                setattr(self, f"_should_compute_{t.key}", True)
            self._should_compute_identity = True
            self._should_compute_translation = True
            self.update_task_schedule = self.parallel_schedule

        elif self.opt.task_schedule in {"sequential", "liss"}:
            for t in self.tasks:
                setattr(self, f"_should_compute_{t.key}", False)
            setattr(self, f"_should_compute_{self.tasks.keys[0]}", True)
            self._should_compute_identity = False
            self._should_compute_translation = False
            if self.opt.task_schedule == "liss":
                self.update_task_schedule = self.liss_schedule
            else:
                self.update_task_schedule = self.sequential_schedule

        elif self.opt.task_schedule in {"additional"}:
            for t in self.tasks:
                setattr(self, f"_should_compute_{t.key}", False)
            setattr(self, f"_should_compute_{self.tasks.keys[0]}", True)
            self._should_compute_identity = False
            self._should_compute_translation = False
            self.update_task_schedule = self.additional_schedule

        elif self.opt.task_schedule == "representational-traduction":
            for t in self.tasks:
                setattr(self, f"_should_compute_{t.key}", True)
            self._should_compute_identity = True
            self._should_compute_translation = True
            self.repr_is_frozen = False
            self.update_task_schedule = self.representational_traduction_schedule
            if self.is_dp:
                freeze = [self.netG_A.module.decoder, self.netG_B.module.decoder]
            else:
                freeze = [self.netG_A.decoder, self.netG_B.decoder]
            freeze += [self.netD_A, self.netD_B]
            self.set_requires_grad(freeze, requires_grad=False)

        elif self.opt.task_schedule == "representational":
            for t in self.tasks:
                setattr(self, f"_should_compute_{t.key}", True)
            self._should_compute_identity = False
            self._should_compute_translation = False
            self.repr_is_frozen = False
            self.update_task_schedule = self.representational_schedule

        else:
            raise ValueError("Unknown schedule {}".format(self.opt.task_schedule))

    def no_grad(self, has_D=False):
        raise NotImplementedError("using no_grad function which is deprecated")
        models = [self.netG_A, self.netG_B]
        if has_D:
            models += [self.netD_A, self.netD_B]
        self.set_requires_grad(models, False)

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights;
        called in every training iteration"""
        # forward
        self.forward()  # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad(
            [
                self.get(d)
                for d in dir(self)
                if hasattr(self, d) and d.startswith("netD_")
            ],
            False,
        )  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()  # calculate gradients for G_A and G_B
        self.optimizer_G.step()  # update G_A and G_B's weights
        # D_A and D_B
        self.optimizer_D.zero_grad()  # set D_A and D_B's gradients to zero
        self.backward_D()  # set_requires_grad in there
        self.optimizer_D.step()

    def get(self, key):
        return getattr(self, key)

    def set(self, key, value):
        setattr(self, key, value)
