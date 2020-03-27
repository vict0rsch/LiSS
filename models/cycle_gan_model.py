import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks


class CycleGANModel(BaseModel):
    """
    This class implements the CycleGAN model, for learning image-to-image translation without paired data.
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
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.
        Returns:
            the modified parser.
        For CycleGAN, in addition to GAN losses, we introduce lambda_A, lambda_B, and lambda_identity for the following losses.
        A (source domain), B (target domain).
        Generators: G_A: A -> B; G_B: B -> A.
        Discriminators: D_A: G_A(A) vs. B; D_B: G_B(B) vs. A.
        Forward cycle loss:  lambda_A * ||G_B(G_A(A)) - A|| (Eqn. (2) in the paper)
        Backward cycle loss: lambda_B * ||G_A(G_B(B)) - B|| (Eqn. (2) in the paper)
        Identity loss (optional): lambda_identity * (||G_A(B) - B|| * lambda_B + ||G_B(A) - A|| * lambda_A) (Sec 5.2 "Photo generation from paintings" in the paper)
        Dropout is not used in the original CycleGAN paper.
        """
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument(
                "--lambda_A",
                type=float,
                default=10.0,
                help="weight for cycle loss (A -> B -> A)",
            )
            parser.add_argument(
                "--lambda_B",
                type=float,
                default=10.0,
                help="weight for cycle loss (B -> A -> B)",
            )
            parser.add_argument(
                "--lambda_identity",
                type=float,
                default=0.5,
                help="use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1",
            )

        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.
        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = [
            "D_A",
            "G_A",
            "A_cycle",
            "A_idt",
            "D_B",
            "G_B",
            "B_cycle",
            "B_idt",
        ]
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ["A_real", "B_fake", "A_rec"]
        visual_names_B = ["B_real", "A_fake", "B_rec"]
        if self.isTrain and self.opt.lambda_identity > 0.0:
            # if identity loss is used, we also visualize B_idt=G_A(B) ad A_idt=G_A(B)
            visual_names_A.append("B_idt")
            visual_names_B.append("A_idt")

        # combine visualizations for A and B
        self.visual_names = visual_names_A + visual_names_B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ["G_A", "G_B", "D_A", "D_B"]
        else:  # during test time, only load Gs
            self.model_names = ["G_A", "G_B"]

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

        if self.isTrain:
            if (
                opt.lambda_identity > 0.0
            ):  # only works when input and output images have the same number of channels
                assert opt.input_nc == opt.output_nc
            self.A_fake_pool = ImagePool(
                opt.pool_size
            )  # create image buffer to store previously generated images
            self.B_fake_pool = ImagePool(
                opt.pool_size
            )  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(
                self.device
            )  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(
                [
                    {
                        "params": itertools.chain(
                            self.netG_A.parameters(), self.netG_B.parameters()
                        ),
                        "lr": opt.lr,
                        "group_name": "default",
                    }
                ],
                betas=(opt.beta1, 0.999),
            )
            self.optimizer_D = torch.optim.Adam(
                [
                    {
                        "params": itertools.chain(
                            self.netD_A.parameters(), self.netD_B.parameters()
                        ),
                        "lr": opt.lr,
                        "group_name": "default",
                    }
                ],
                betas=(opt.beta1, 0.999),
            )
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == "AtoB"
        self.A_real = input["A" if AtoB else "B"].to(self.device)
        self.B_real = input["B" if AtoB else "A"].to(self.device)
        self.image_paths = input["A_paths" if AtoB else "B_paths"]

    def forward(self, ignore=None, force=None):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        lambda_idt = self.opt.lambda_identity
        self.B_fake = self.netG_A(self.A_real)  # G_A(A)
        self.A_rec = self.netG_B(self.B_fake)  # G_B(G_A(A))
        self.A_fake = self.netG_B(self.B_real)  # G_B(B)
        self.B_rec = self.netG_A(self.A_fake)  # G_A(G_B(B))
        if lambda_idt > 0:
            self.A_idt = self.netG_A(self.B_real)
            self.B_idt = self.netG_B(self.A_real)

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator
        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator
        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        B_fake = self.B_fake_pool.query(self.B_fake)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.B_real, B_fake)

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        A_fake = self.A_fake_pool.query(self.A_fake)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.A_real, A_fake)

    def backward_G(self, losses_only=False, force=None):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if B_real is fed: ||G_A(B) - B||

            self.loss_A_idt = (
                self.criterionIdt(self.A_idt, self.B_real) * lambda_B * lambda_idt
            )
            # G_B should be identity if A_real is fed: ||G_B(A) - A||
            self.loss_B_idt = (
                self.criterionIdt(self.B_idt, self.A_real) * lambda_A * lambda_idt
            )
        else:
            self.loss_A_idt = 0
            self.loss_B_idt = 0

        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.B_fake), True)
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.A_fake), True)
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_A_cycle = self.criterionCycle(self.A_rec, self.A_real) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_B_cycle = self.criterionCycle(self.B_rec, self.B_real) * lambda_B
        # combined loss and calculate gradients
        self.loss_G = (
            self.loss_G_A
            + self.loss_G_B
            + self.loss_A_cycle
            + self.loss_B_cycle
            + self.loss_A_idt
            + self.loss_B_idt
        )
        if not losses_only:
            self.loss_G.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()  # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad(
            [self.netD_A, self.netD_B], False
        )  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()  # calculate gradients for G_A and G_B
        self.optimizer_G.step()  # update G_A and G_B's weights
        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()  # set D_A and D_B's gradients to zero
        self.backward_D_A()  # calculate gradients for D_A
        self.backward_D_B()  # calculate gradients for D_B
        self.optimizer_D.step()  # update D_A and D_B's weights

    def get(self, key):
        return getattr(self, key)

    def update_task_schedule(self, metrics=None):
        pass
