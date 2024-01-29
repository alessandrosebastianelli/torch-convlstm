from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.model_selection import ParameterGrid
import pytorch_lightning as pl
import numpy as np
import torch
import sys
import os

sys.path += ['.', './']

from models.ConvLSTM import EncoderDecoderConvLSTM
from dataio.loader import SimulatedCloudsDataset

if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')

    future_seq = 4

    data_module = SimulatedCloudsDataset(batch_size=16, num_workers=4, in_lenght=8, ou_lenght=future_seq)
    tb_logger = pl.loggers.TensorBoardLogger(os.path.join('lightning_logs','convlstm'), name='EncoderDecoderConvLSTM')

    # Instantiate ModelCheckpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join('saved_models','EncoderDecoderConvLSTM'),
        filename='EncoderDecoderConvLSTM',
        monitor='valid_loss',
        save_top_k=1,
        mode='min',
    )

    # Instantiate LightningModule and DataModule
    model = EncoderDecoderConvLSTM(nf=32, in_chan=1, future_seq=future_seq)

    # Instantiate Trainer
    trainer = pl.Trainer(max_epochs=50, callbacks=[checkpoint_callback], logger=tb_logger)

    # Train the model
    trainer.fit(model, data_module)
