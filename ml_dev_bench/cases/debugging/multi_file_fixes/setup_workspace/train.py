import json
import logging
import os

import wandb
from cifar10_classifier.data import CIFAR10DataModule
from cifar10_classifier.model import CIFAR10Classifier
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger


def main():
    # Configure logging
    logging.basicConfig(
        filename='training.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
    )

    # Initialize WandB
    wandb.login(key=os.getenv('WANDB_API_KEY'))
    wandb_logger = WandbLogger(project='cifar10-training', log_model=True)

    # Initialize data module
    data_module = CIFAR10DataModule(batch_size=32)

    # Initialize model
    model = CIFAR10Classifier()

    # Set up callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, mode='min')

    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints',
        filename='cifar10-{epoch:02d}-{val_loss:.2f}',
        monitor='val_loss',
        mode='min',
        save_top_k=1,
    )

    # Initialize trainer
    trainer = Trainer(
        max_epochs=2,
        accelerator='cpu',  # Use CPU to avoid CUDA issues
        logger=wandb_logger,
        callbacks=[early_stopping, checkpoint_callback],
    )

    # Prepare data
    data_module.prepare_data()
    data_module.setup()

    # Train the model
    trainer.fit(model, datamodule=data_module)

    # Save WandB info
    wandb_info = {
        'project_id': wandb_logger.experiment.project,
        'run_id': wandb_logger.experiment.id,
        'run_url': wandb_logger.experiment.url,
    }

    with open('wandb_info.json', 'w') as f:
        json.dump(wandb_info, f, indent=4)

    wandb.finish()


if __name__ == '__main__':
    main()
