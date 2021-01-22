import argparse
import importlib
import os

from pytorch_lightning import Trainer, loggers
from pytorch_lightning.callbacks import ModelCheckpoint


def parse_args():
    parser = argparse.ArgumentParser(description='Input Experiment Settings')
    parser.add_argument('--model', required=True, type=str)
    parser.add_argument('--config', required=True, type=str)
    parser.add_argument('--dataset', required=True, type=str)
    parser.add_argument('--option', default='Default', type=str)

    args = parser.parse_args()
    return args


def train():
    args = parse_args()

    model_module = importlib.import_module('models.' + args.model + '.' + args.model)
    config_module = importlib.import_module('configs.' + args.model + '.' + args.config)
    data_module = importlib.import_module('datasets.' + args.dataset)

    model_cls = getattr(model_module, args.model)
    hparams = config_module.get_hparams(args.option)
    dataset_cls = getattr(data_module, args.dataset+'DataModule')
    data_hparams = config_module.get_data_hparams(args.option)

    model = model_cls(hparams)
    dataset = dataset_cls(data_hparams)

    tb_logger = loggers.TensorBoardLogger(
        save_dir=os.path.join('logs', args.model+'_logs'),
        name=args.config+'_'+args.option
    )

    checkpoint_callback = ModelCheckpoint(
        save_top_k=-1
    )

    weights_save_path = os.path.join('logs', args.model + '_logs')

    trainer = Trainer(gpus=-1, logger=tb_logger,
                      checkpoint_callback=checkpoint_callback,
                      log_every_n_steps=250,
                      weights_save_path=weights_save_path,
                      distributed_backend='ddp',
                      replace_sampler_ddp=False)

    trainer.fit(model, dataset)


if __name__ == '__main__':
    train()
