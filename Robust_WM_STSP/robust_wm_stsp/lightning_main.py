from lightning_networks import dDMTSNet
from lightning_task import dDMTSDataModule
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import argparse
from pytorch_lightning import Trainer
from pytorch_lightning.plugins import DDPPlugin
import os

input_size = 8 + 3
#hidden_size = 100
output_size = 8 + 3
dt_ann = 15
alpha = dt_ann / 100
alpha_W = dt_ann / 100
g = 0.9

AVAIL_GPUS = max(1, torch.cuda.device_count())

BATCH_SIZE = 256 if AVAIL_GPUS else 32
#BATCH_SIZE = 64

if __name__ == "__main__":

    print('The number of availible GPUS is: ' + str(AVAIL_GPUS))

    parser = argparse.ArgumentParser(description="rnn dDMTS")

    
    #network parameters & hyperparameters
    parser.add_argument("--rnn_type", type=str, default="vRNN", help="rNN to use")   
    parser.add_argument("--nl", type=str, default="tanh", help="nonlinearity to use")  
    parser.add_argument("--hs", type=int, default="100", help="hidden size")  
    parser.add_argument("--gamma", type=float, default=".005", help="leak rate of anti-hebbian plasticity")  
     

    #learning hyperparameters
    parser.add_argument("--lr", type=float, default="1e-3", help="learning rate to use")
    parser.add_argument("--act_reg", type=float, default="1e-3", help="activity regularization strength") 
    parser.add_argument("--param_reg", type=float, default="1e-4", help="parameter regularization strength")
    parser.add_argument("--epochs", type=int, default=10, help="number of epochs to train (default: 10)")
    
    
    
    
    args = parser.parse_args()

    checkpoint_callback = ModelCheckpoint(
        monitor="val_acc",
        dirpath="_lightning_sandbox/checkpoints/",
        filename="rnn-sample-dDMTS-{epoch:02d}-{val_acc:.2f}--"
        + args.rnn_type
        + "--"
        + args.nl,
        every_n_epochs=args.epochs,
        mode="max",
        save_last=True,
    )
    
    '''
    checkpoint_callback.CHECKPOINT_NAME_LAST = ("last_rnn-sample-dDMTS-{epoch:02d}-{val_acc:.2f}--" + args.rnn_type    + "--"
    + args.nl
    )
    '''
    
    checkpoint_callback.CHECKPOINT_NAME_LAST = (f"rnn={args.rnn_type}--nl={args.nl}--hs={args.hs}--act_reg={args.act_reg}--gamma={args.gamma}--param_reg={args.param_reg}--" + "{epoch:02d}--{val_acc:.2f}")
    






    early_stop_callback = EarlyStopping(
        monitor="val_acc", stopping_threshold=0.95, mode="max", patience=50
    )

    model = dDMTSNet(
        args.rnn_type,
        input_size,
        args.hs,
        output_size,
        dt_ann,
        alpha,
        alpha_W,
        g,
        args.nl,
        args.lr,
    )

    model.act_reg = args.act_reg
    model.param_reg = args.param_reg
    
    if args.rnn_type == 'ah':
        model.rnn.gamma_val = args.gamma



    dDMTS = dDMTSDataModule(dt_ann=dt_ann)


    trainer = Trainer(
        max_epochs=args.epochs,
        progress_bar_refresh_rate=20,
        callbacks=[checkpoint_callback, early_stop_callback],
        accelerator="ddp",
        log_every_n_steps=10,
        plugins=DDPPlugin(find_unused_parameters=False),
        gpus = [1,2]
    )

    trainer.fit(model, dDMTS)

    # trainer.save_checkpoint("example.ckpt")

