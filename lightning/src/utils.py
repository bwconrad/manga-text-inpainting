import pprint
import os
import yaml
from argparse import Namespace
from torch.optim import lr_scheduler

def get_scheduler(optimizer, hparams):
    name = hparams.schedule

    if name == 'cosine':
        print('Using Cosine LR schedule')
        return {
            'scheduler': lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=hparams.epochs,
            ),
        }
    elif name == 'step':
        print(f'Using Step LR scheule with steps={hparams.steps} and step_size={hparams.step_size}')
        return {
            'scheduler': lr_scheduler.MultiStepLR(
                    optimizer,
                    milestones=hparams.steps,
                    gamma=hparams.step_size,
            ),
        }
    elif name == 'none':
        print('Using no LR schedule')
        return {
            'scheduler': lr_scheduler.LambdaLR(
                    optimizer,
                    lr_lambda=lambda epoch: 1,
            ),
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


default_hparams = {}