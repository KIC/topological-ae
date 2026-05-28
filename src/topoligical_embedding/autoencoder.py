import torch
import torch.nn as nn
import numpy as np
import warnings
warnings.filterwarnings("ignore", message="Provided metric name", module="ignite")

from pathlib import Path
from ignite.handlers import ProgressBar
from dotenv import load_dotenv
from torch_topological.nn import SignatureLoss, VietorisRipsComplex
from ignite.handlers.wandb_logger import WandBLogger, OutputHandler, OptimizerParamsHandler
from ignite.handlers import Checkpoint, DiskSaver, EarlyStopping
from ignite.engine import Engine, Events
from ignite.metrics import Average
from torch.utils.data import Dataset, DataLoader
from sklearn.manifold import trustworthiness

load_dotenv()


class Autoencoder(nn.Module):
    def __init__(self, input_dim=21, latent_dim=5, hidden_dims=(64, 64, 32, 16), use_bn=False):
        super().__init__()
        self._conf = dict(input_dim=input_dim, latent_dim=latent_dim, hidden_dims=hidden_dims, use_bn=use_bn)

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
    def __init__(
        self,
        training_data: str | Path, 
        training_labels: str | Path | None = None, 
        mmap_mode: bool = False,
        device: torch.device | None = None
    ):
        self.features = torch.from_numpy(np.load(training_data, mmap_mode=('r+' if mmap_mode else None)))
        self.labels = torch.from_numpy(np.load(training_labels)) if training_labels is not None else None
        self.device = device if not mmap_mode else None

        if device and not mmap_mode:
            self.features = self.features.to(device)
            self.labels = self.labels.to(device) if self.labels is not None else None

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        features = self.features[idx]
        labels = self.labels[idx] if self.labels is not None else features

        if self.device:
            features = features.to(self.device)
            if self.labels:
                labels = labels.to(self.device)

        return features, labels


def create_topo_train_step(model, optimizer, lam, device, noise=None, report_trustworthiness=False, is_training=True):
    topo_lossfn = SignatureLoss(p=2)
    mse_lossfn = torch.nn.MSELoss()
    similarity = nn.CosineSimilarity()
    vr = VietorisRipsComplex(dim=0)
    
    def train_step(engine, batch):
        if is_training:
            model.train()
            optimizer.zero_grad()
        else:
            model.eval()
        
        x, _ = batch
        x = x.to(device)
        
        # Add noise to features, should make it more stable
        x_noisy = (x + noise * torch.randn_like(x)) if noise is not None else x
        
        z, x_hat = model(x_noisy)
        
        pi_x = vr(x)
        pi_z = vr(z)

        topo_loss = topo_lossfn([x, pi_x], [z, pi_z])
        mse_loss = mse_lossfn(x, x_hat)
        direction_loss = torch.mean(1 - similarity(x, x_hat))
        
        total_loss = mse_loss + direction_loss + (lam * topo_loss)
        
        if is_training:
            total_loss.backward()
            optimizer.step()

        tw = 0.0
        if report_trustworthiness:
            with torch.no_grad():
                tw = trustworthiness(x.cpu().numpy(), z.cpu().numpy())
        
        return {
            "total_loss": total_loss.item(),
            "mse_loss": mse_loss.item(),
            "direction_loss": direction_loss.item(),
            "topo_loss": topo_loss.item(),
            "trustworthiness": tw,
        }
        
    return train_step


def load_checkpoint(
    checkpoint_path: str | Path,
    model: Autoencoder,
    optimizer: torch.optim.Optimizer,
    trainer: Engine,
) -> None:
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    Checkpoint.load_objects(
        to_load={"model": model, "optimizer": optimizer, "trainer": trainer},
        checkpoint=checkpoint,
    )


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    noise=0.05
    lam=1.0
    batch_size = 1024 * 2
    learning_rate = 1e-3 / 2
    mmap_mode=False
    
    model = Autoencoder(use_bn=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    wandb_logger = WandBLogger(
        project="topological-autoencoder",
        config={"learning_rate": learning_rate, "lam": lam, "batch_size": batch_size, "noise": noise, "model": model._conf}
    )

    trainer = Engine(create_topo_train_step(model, optimizer, lam=lam, device=device, noise=noise, is_training=True, report_trustworthiness=True))
    evaluator = Engine(create_topo_train_step(model, optimizer, lam=lam, device=device, is_training=False, report_trustworthiness=True))

    print("Loading data...")
    train_loader = DataLoader(
        NumpyDataset(
            'data/vzscores32.npy.weighted.npy.shuffled.npy.train.npy',
            #'data/vzscores32.npy.weighted.npy.shuffled.npy.train.npy.shuffled.npy',
            #device=device,
            mmap_mode=True
        ),
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
    )

    test_loader = DataLoader(
        NumpyDataset(
            'data/vzscores32.npy.weighted.npy.shuffled.npy.test.npy', 
            #'data/vzscores32.npy.weighted.npy.shuffled.npy.test.npy.shuffled.npy',
            #device=device, 
            mmap_mode=True,
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
    )

    print("Starting training...")
    metrics = ["total_loss", "mse_loss", "direction_loss", "topo_loss", "trustworthiness"]
    for metric_name in metrics:
        Average(output_transform=lambda x, m=metric_name: x[m]).attach(trainer, metric_name)
        Average(output_transform=lambda x, m=metric_name: x[m]).attach(evaluator, metric_name)

    ProgressBar(persist=True).attach(trainer, metric_names=metrics)
    ProgressBar(persist=False, desc="Validation").attach(evaluator, metric_names=metrics)

    wandb_logger.attach(
        trainer,
        log_handler=OptimizerParamsHandler(optimizer),
        event_name=Events.EPOCH_COMPLETED
    )

    wandb_logger.attach(
        trainer,
        log_handler=OutputHandler(
            tag="train",
            metric_names=metrics,
            global_step_transform=lambda engine, event: engine.state.epoch,
        ),
        event_name=Events.EPOCH_COMPLETED
    )

    wandb_logger.attach(
        evaluator,
        log_handler=OutputHandler(
            tag="val",
            metric_names=metrics,
            global_step_transform=lambda engine, event: engine.state.epoch,
        ),
        event_name=Events.EPOCH_COMPLETED
    )

    @trainer.on(Events.EPOCH_COMPLETED)
    def run_test_set(engine):
        with torch.no_grad():            
            evaluator.run(test_loader)   

    @trainer.on(Events.COMPLETED)
    def end_wandb(engine):
        # calls wandb.finish internally
        wandb_logger.close()

    @evaluator.on(Events.EPOCH_COMPLETED)
    def log_test_results(engine):
        print(f"\n--- Epoch {trainer.state.epoch} Test Metrics ---")
        for name, value in engine.state.metrics.items():
            print(f"Test {name}: {value:.4f}")
        print("-" * 30 + "\n")

    # Configure a saver that writes to disk
    checkpoint_handler = Checkpoint(
        {"model": model, "optimizer": optimizer, "trainer": trainer},
        DiskSaver(dirname="./snapshots", create_dir=True, require_empty=False),
        n_saved=10,
        filename_prefix="epoch_",
        score_function=None,
    )

    # Attach to trainer so it runs after each epoch
    trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler)

    # Add early stopping
    early_stopping = EarlyStopping(
        patience=3,
        score_function=lambda engine: -engine.state.metrics["total_loss"],
        trainer=trainer,
    )
    evaluator.add_event_handler(Events.COMPLETED, early_stopping)

    resume_from = None  # set to a snapshot path to resume, e.g. "./snapshots/epoch__checkpoint_3.pt"
    if resume_from:
        load_checkpoint(resume_from, model, optimizer, trainer)
        print(f"Resumed from {resume_from} (epoch {trainer.state.epoch})")

    trainer.run(train_loader, max_epochs=13)
