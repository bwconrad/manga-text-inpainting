import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
import os
from pathlib import Path
from argparse import ArgumentParser, Namespace

from src.data_module import MaskRefineDataModule
from src.utils import hparams_from_config
from src.model import MaskRefineModel

parser = ArgumentParser()
parser.add_argument('-c', '--config_path', type=Path, help='Path to the config.', default='.') 
parser = pl.Trainer.add_argparse_args(parser)
args = parser.parse_args()

# Load hparams from config file
hparams = hparams_from_config(args.config_path)
args.max_epochs = hparams.epochs

# Define callbacks
tb_logger = TensorBoardLogger(
    save_dir=hparams.output_path,
    name=hparams.experiment_name
)
csv_logger = CSVLogger(
    save_dir=hparams.output_path,
    name=hparams.experiment_name
)
checkpoint_callback = ModelCheckpoint(
    filepath=os.path.join(tb_logger.root_dir, 'best-{epoch}-{val_f1:.4f}'),
    save_top_k=1,
    verbose=True,
    monitor='val_f1',
    mode='max',
    save_last=True,
)

# Load datamodule
dm = MaskRefineDataModule(hparams)

# Load model
model = MaskRefineModel(hparams)

# Run trainer
trainer = pl.Trainer.from_argparse_args(
    args,
    checkpoint_callback=checkpoint_callback,
    logger=[tb_logger, csv_logger],
)

trainer.fit(model, dm)