import torch
# from .base_model import BaseTrainer
import network
import util
from util import timeit
from config import cfg
import torchvision

def mk_models():

    netG = network.define_G(
        cfg.inc,
        cfg.onc,
        cfg.ngf,
        cfg.netG,
        cfg.norm,
        not cfg.no_dropout,
        cfg.init_type,
        cfg.init_gain,
        cfg.gpu_ids,
    )

    netD = network.define_D(
        cfg.inc + cfg.onc,
        cfg.ndf,
        cfg.netD,
        cfg.n_layers_D,
        cfg.norm,
        cfg.init_type,
        cfg.init_gain,
        cfg.gpu_ids,
    )
    return netG, netD



def mk_dataset():
    """makes a dataset"""
    return torchvision.datasets.Cityscapes(
        root="/grand/EVITA/datasets/cityscapes",
        split="train",
        mode="fine",
        target_type="semantic",
    )



class CycleGanTrainer(): # (BaseTrainer)
    def __init__(self):
        """Initialize the pix2pix class.

        Parameters:
            cfg (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """

        # super(self, Pix2PixTrainer).__init__()

        self.loss_names = ["G_GAN", "G_L1", "D_real", "D_fake"]
        self.visual_names = ["real_A", "fake_B", "real_B"]

        self.netG, self.netD = mk_models()

        self.device = cfg.gpu_ids[0]
        self.criterionGAN = util.GANLoss(cfg.gan_mode).to(self.device)
        self.criterionL1 = torch.nn.L1Loss()

        opt_kwargs = dict(lr=cfg.lr, betas=(cfg.beta1, 0.999))
        mk_opt = lambda param: torch.optim.Adam(param, **opt_kwargs)
        self.optG = mk_opt(self.netG.parameters())
        self.optD = mk_opt(self.netD.parameters())
        self.optimizers = [ self.optG, self.optD]

        dataset = mk_dataset()
        dataset_size = len(dataset)
        print(f"The number of training images = {dataset_size}")

        model.setup(opt)  # regular setup: load and print network; create schedulers

        self.vis = Visualizer(opt)  # create a visualizer that display/save images and plots

        self.nepoch, self.niter = 0,0


    def unpack(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.cfg.direction == "AtoB"
        self.real_A = input["A" if AtoB else "B"].to(self.device)
        self.real_B = input["B" if AtoB else "A"].to(self.device)
        self.image_paths = input["A_paths" if AtoB else "B_paths"]

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG(self.real_A)  # G(A)

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        # we use conditional GANs; we need to feed both input and output to the discriminator
        fake_AB = torch.cat( (self.real_A, self.fake_B), 1)  
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)

        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)

        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""

        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)

        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.cfg.lambda_L1

        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()  # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optD.zero_grad()  # set D's gradients to zero
        self.backward_D()  # calculate gradients for D
        self.optD.step()  # update D's weights
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optG.zero_grad()  # set G's gradients to zero
        self.backward_G()  # calculate graidents for G
        self.optG.step()  # update G's weights

    def housekeeping():
        """docstring"""

        if self.niter % cfg.display_freq == 0:
            save_result = self.niter % cfg.update_html_freq == 0
            self.compute_visuals()
            self.vis.display_current_results(self.get_current_visuals(), self.nepoch, save_result)

        if self.niter % cfg.print_freq == 0:
            losses = self.get_current_losses()
            self.vis.print_current_losses( self.nepoch, self.niter, losses, t_comp, t_data)  # time comp time data
            if cfg.display_id > 0:
                self.vis.plot_current_losses(self.nepoch, float(self.niter) / dataset_size, losses)

        if self.niter % cfg.save_latest_freq == 0:
            print(f"saving the latest model (epoch {self.nepoch}, total_iters {self.niter })")
            save_suffix = f"iter_{self.niter if cfg.save_by_iter else 'latest'}"
            self.save_networks(save_suffix)



    @timeit
    def step():
        """docstring"""

        self.niter += 1

        self.unpack(data)
        self.optimize_parameters()  # calculate loss functions, get gradients, update network weights
        self.housekeeping()

    @timeit
    def epoch():
        """docstring"""

        self.nepoch = 0
        self.vis.reset()  # reset the visualizer: make sure it saves the results to HTML at least once every epoch
        self.update_learning_rate()

        for i, data in enumerate(dataset):
            step()

        if self.nepoch % cfg.save_epoch_freq == 0:
            print(f"saving the model at the end of epoch {self.nepoch}, iters {total_iters})")
            self.save_networks("latest")
            self.save_networks(self.nepoch)

    @timeit
    def run():
        """docstring"""

        total_iters = 0
        for _ in range(cfg.nepoch, cfg.n_epochs + cfg.n_epochs_decay + 1):
            epoch()



def main():
    trainer = Pix2PixTrainer()
    trainer.run()


if __name__ == "__main__":
    main()
