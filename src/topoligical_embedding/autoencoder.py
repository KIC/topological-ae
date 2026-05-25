from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

from dotenv import load_dotenv
from torch_topological.nn import SignatureLoss, VietorisRipsComplex
from ignite.handlers.wandb_logger import WandBLogger, OutputHandler, OptimizerParamsHandler
from ignite.engine import Engine, Events
from ignite.metrics import Average
from torch.utils.data import Dataset, DataLoader
from sklearn.manifold import trustworthiness

load_dotenv()


class Autoencoder(nn.Module):
    def __init__(self, input_dim=20, latent_dim=5, hidden_dims=(64, 32, 16), use_bn=False):
        super().__init__()

        enc_layers = []
        prev = input_dim
        for h in hidden_dims:
            enc_layers.append(nn.Linear(prev, h))
            if use_bn:
                enc_layers.append(nn.BatchNorm1d(h))
            enc_layers.append(nn.PReLU())
            prev = h
        enc_layers.append(nn.Linear(prev, latent_dim))
        self.encoder = nn.Sequential(*enc_layers)

        dec_layers = []
        prev = latent_dim
        for h in reversed(hidden_dims):
            dec_layers.append(nn.Linear(prev, h))
            if use_bn:
                dec_layers.append(nn.BatchNorm1d(h))
            dec_layers.append(nn.PReLU())
            prev = h
        dec_layers.append(nn.Linear(prev, input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x, return_latent=False):
        z = self.encoder(x)
        if return_latent:
            return z
        
        return z, self.decoder(z) 



class NumpyDataset(Dataset):
    def __init__(self, training_data: str | Path, training_labels: str | Path | None = None, mmap_mode: bool = False):
        self.features = np.load(training_data, mmap_mode=('r+' if mmap_mode else None))
        self.labels = np.load(training_labels) if training_labels is not None else None

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        features = self.features[idx]
        return features, self.labels[idx] if self.labels is not None else features


def create_topo_train_step(model, optimizer, lam, device, report_trustworthiness=False, is_training=True):
    topo_lossfn = SignatureLoss(p=2)
    mse_lossfn = torch.nn.MSELoss()
    similarity = nn.CosineSimilarity()
    vr = VietorisRipsComplex(dim=0)
    
    # This is the actual closure function Ignite will run
    def train_step(engine, batch):
        model.train()
        optimizer.zero_grad()
        
        x, _ = batch
        x = x.to(device)
        
        # 1. Forward pass returns latent (z) and reconstruction (x_hat)
        z, x_hat = model(x)
        
        pi_x = vr(x)
        pi_z = vr(z)

        # 2. Compute individual error fragments directly in the step
        topo_loss = topo_lossfn([x, pi_x], [z, pi_z])
        mse_loss = mse_lossfn(x, x_hat)
        direction_loss = torch.mean(1 - similarity(x, x_hat))
        
        # 3. Compose total loss
        total_loss = mse_loss + direction_loss + (lam * topo_loss)
        
        # 4. Optimization steps
        if is_training:
            total_loss.backward()
            optimizer.step()

        if report_trustworthiness:
            with torch.no_grad():
                tw = trustworthiness(x, z)
        else:
            tw = 0
        
        # Return a dictionary of components for Ignite to monitor
        return {
            "total_loss": total_loss.item(),
            "mse_loss": mse_loss.item(),
            "direction_loss": direction_loss.item(),
            "topo_loss": topo_loss.item(),
            "trustworthiness": tw,
        }
        
    return train_step


if __name__ == '__main__':
    # --- 2. INITIALIZE COMPONENTS ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Autoencoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    batch_size = 1024

    wandb_logger = WandBLogger(
        project="topological-autoencoder",
        name="ignite-run",
        config={"learning_rate": 1e-3, "lam": 1.0}
    )

    # Initialize your structural and topological loss functions here and create the engine
    trainer = Engine(create_topo_train_step(model, optimizer, lam=1.0, device=device))
    evaluator = Engine(create_topo_train_step(model, optimizer, lam=1.0, device=device, is_training=False))

    # create data loaders
    train_loader = DataLoader(
        NumpyDataset('/home/badger/data/sources/mine/quant/TorchSOM-1.1.1/pumap/vzscores32.npy.weighted.npy.shuffled.npy.train.npy'),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )

    test_loader = DataLoader(
        NumpyDataset('/home/badger/data/sources/mine/quant/TorchSOM-1.1.1/pumap/vzscores32.npy.weighted.npy.shuffled.npy.test.npy'),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False
    )

    # Attach running average metrics to the engine 
    metrics = ["total_loss", "mse_loss", "direction_loss", "topo_loss", "trustworthiness"]
    for metric_name in metrics:
        Average(output_transform=lambda x, m=metric_name: x[m]).attach(trainer, metric_name)
        Average(output_transform=lambda x, m=metric_name: x[m]).attach(evaluator, metric_name)

    # 4. Attach the logger to track metrics every iteration (or change to Events.EPOCH_COMPLETED)
    wandb_logger.attach(
        trainer,
        log_handler=OutputHandler(
            tag="training",
            metric_names=metrics # This reads the attached metrics from step 2
        ),
        event_name=Events.ITERATION_COMPLETED
    )

    # Optional: Track the learning rate over time
    wandb_logger.attach(
        trainer,
        log_handler=OptimizerParamsHandler(optimizer),
        event_name=Events.ITERATION_COMPLETED
    )


    # 3. Connect the test engine to WandB (just like the trainer)
    wandb_logger.attach(
        evaluator,
        log_handler=OutputHandler(
            tag="test",
            metric_names=metrics,
            global_step_transform=lambda *_: trainer.state.epoch # Plugs neatly into your epoch timeline
        ),
        event_name=Events.EPOCH_COMPLETED
    )

    # Event handlers
    @trainer.on(Events.EPOCH_COMPLETED)
    def run_test_set(engine):
        model.eval()                     # Switch model to evaluation mode
        with torch.no_grad():            # Turn off gradients to save memory and skip training updates
            evaluator.run(test_loader)   # Run the exact same loop over test data


    # Make sure the wandb context closes cleanly when engine finishes
    @trainer.on(Events.COMPLETED)
    def end_wandb(engine):
        wandb_logger.close()


    # Finally train this shit
    trainer.run(train_loader, max_epochs=10)