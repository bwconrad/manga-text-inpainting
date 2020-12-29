import pprint
import os
import yaml
from argparse import Namespace
from torch.optim import lr_scheduler

def get_scheduler(optimizer, lr, hparams):
    name = hparams.schedule

    if name == 'cosine':
        print(f'Using Cosine LR schedule lr={lr}')
        return {
            'scheduler': lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=hparams.epochs,
            ),
        }
    elif name == 'step':
        print(f'Using Step LR schedule with lr={lr} steps={hparams.steps} step_size={hparams.step_size}')
        return {
            'scheduler': lr_scheduler.MultiStepLR(
                    optimizer,
                    milestones=hparams.steps,
                    gamma=hparams.step_size,
            ),
        }
    elif name == 'plateau':
        print(f'Using Plateau LR schedule with lr={lr} factor={hparams.plateau_factor}',
              f'patience={hparams.plateau_patience} # reduces={hparams.plateau_n}')
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=hparams.plateau_patience,
            factor=hparams.plateau_factor,
            verbose=True,
            mode='max',
            min_lr=lr*hparams.plateau_factor**hparams.plateau_n # Allow up to plateau_n reduces
        )
        return {
           'scheduler': scheduler,
           'monitor': 'val_loss',
           'reduce_on_plateau': True, 
           'monitor': 'val_f1',
        }
    else:
        raise NotImplementedError(f'{name} is not an available learning rate schedule')

def hparams_from_config(config_path):
    hparams = default_hparams
    
    if os.path.isfile(config_path):
        # Load config
        with open(config_path) as f:
            config = yaml.load(f, Loader=yaml.SafeLoader)

        # Replace defaults with config file values
        for key, value in config.items():
            hparams[key] = value

    # Print the config file contents
    pp = pprint.PrettyPrinter(indent=4)
    print('Loaded hparams:')
    pp.pprint(hparams)
    return Namespace(**hparams)


default_hparams = {
    'mode': 'mask_refine',      # masked_refine 
    'weights': '',
    'experiment_name': 'test',
    'data_path': 'data/',
    'output_path': 'output',
    'height': 256,
    'width': 256,
    'tversky_alpha': 0.2,       # For mask_refine training
    'tversky_beta': 0.8,        # For mask_refine training
    'arch': 'encoder_decoder',  # Network architecture
    'nf': 32,                   # Number of features channels
    'weight_init': 'kaiming',   # normal | xavier | kaiming | orthogonal
    'weight_gain': 0.02,        # Std for normal, xavier and orthogonal weight init
    'n_blocks': 4,              # Number of resblocks
    'epochs': 1000,
    'lr': 0.001,
    'batch_size': 64,
    'beta1': 0,
    'beta2': 0.9,
    'workers': 6,
    'schedule': 'none',         # none | step | cosine | plateau
    'steps': [],                # Reduction epochs for step schedule
    'step_size': 0.1,           # Reduction factor for step schedule
    'plateau_patience': 4,      # Patience for plateau schedule
    'plateau_factor': 0.1,      # Reduction factor for plateau schedule     
    'plateau_n': 2,             # Number of reductions for plateau schedule
    'early_stop_patience': 10,  # Patience for early stopping ()
}