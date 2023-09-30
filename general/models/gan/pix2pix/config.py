import yacs

from yacs.config import CfgNode as CN

cfg = {
    'dataroot': None,
    'name': 'experiment_name',
    'gpu_ids': [0,1,2,3],
    'checkpoints_dir': './checkpoints',
    'model': 'cycle_gan',
    'inc': 3,
    'onc': 3,
    'ngf': 64,
    'ndf': 64,
    'netD': 'basic',
    'netG': 'resnet_9blocks',
    'n_layers_D': 3,
    'norm': 'instance',
    'init_type': 'normal',
    'init_gain': 0.02,
    'no_dropout': False,
    'dataset_mode': 'unaligned',
    'gan_mode': 'vanilla', # as opposed to what?
    'direction': 'AtoB',
    'serial_batches': False,
    'num_threads': 4,
    'batch_size': 1,
    'load_size': 286,
    'crop_size': 256,
    'max_dataset_size': float("inf"),
    'preprocess': 'resize_and_crop',
    'no_flip': False,
    'display_winsize': 256,
    'epoch': 'latest',
    'load_iter': 0,
    'verbose': False,
    'suffix': '',
    'use_wandb': False,
    'wandb_project_name': 'CycleGAN-and-pix2pix'
}

train = {
    'display_freq': 400,
    'display_ncols': 4,
    'display_id': 1,
    'display_server': "http://localhost",
    'display_env': 'main',
    'display_port': 8097,
    'update_html_freq': 1000,
    'print_freq': 100,
    'no_html': False,
    'save_latest_freq': 5000,
    'save_epoch_freq': 5,
    'save_by_iter': False,
    'continue_train': False,
    'epoch_count': 1,
    'phase': 'train',
    'n_epochs': 100,
    'n_epochs_decay': 100,
    'beta1': 0.5,
    'lr': 0.0002,
    'gan_mode': 'lsgan',
    'pool_size': 50,
    'lr_policy': 'linear',
    'lr_decay_iters': 50
}



cfg = CN(
    new_allowed=True,
    init_dict={**cfg,**train},
)
