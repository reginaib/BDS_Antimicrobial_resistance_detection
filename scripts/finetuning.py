import torch
import wandb

from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from lightning.pytorch.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from model import MALDIMLP


def initialize_model(config, logger):
    model = MALDIMLP(input_dim=config.input_dim,
                     hidden_dim=config.hidden_dim, 
                     n_hidden_layers=config.n_hidden_layers,
                     lr=config.lr, 
                     dr=config.dr)
    
    monitor = 'Validation/val_loss'
    early_stop_callback = EarlyStopping(monitor=monitor,
                                        min_delta=0.00,
                                        patience=15,
                                        verbose=True,
                                        mode='min')

    trainer = Trainer(accelerator=config.accelerator,
                      max_epochs=config.max_epochs,
                      logger=logger,
                      callbacks=early_stop_callback)

    train_loader = DataLoader(torch.load(config.train_dataset), batch_size=32, shuffle=True)
    valid_loader = DataLoader(torch.load(config.valid_dataset), batch_size=32)
    test_loader = DataLoader(torch.load(config.test_dataset), batch_size=32)
    
    trainer.fit(model, train_loader, valid_loader)
    trainer.test(model, test_loader)

    if config.save_preds:
        predictions = torch.cat(trainer.predict(model, test_loader))
        torch.save(predictions, f'./results/{config.preds_name}')


def optimize_sweep():
    wandb.init()
    config = wandb.config
    logger = WandbLogger()
    initialize_model(config=config, logger=logger)


def start_sweep(config, project_name, num_config=15):
    wandb.login(key='fd8f6e44f8d81be3a652dbd8f4a47a7edf59e44c')
    sweep_id = wandb.sweep(config, project=project_name)
    wandb.agent(sweep_id=sweep_id, function=optimize_sweep, count=num_config)
    

def start_training(config, project_name):
    wandb.login(key='fd8f6e44f8d81be3a652dbd8f4a47a7edf59e44c')
    logger = WandbLogger(project=project_name, job_type='train', log_model='all')
    initialize_model(config, logger)