import torch

from topoligical_embedding.autoencoder import Autoencoder
from topoligical_embedding.trainer import create_topo_train_step


if __name__ == "__main__":
    lam=1.0
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = Autoencoder(use_bn=True)
    
    evaluator = create_topo_train_step(model, None, lam=lam, device=device, is_training=False, report_trustworthiness=True)

    # we need to execute the forward pass and eventually evaluate the loss
    # wandb: Run summary:
    # wandb:    val/direction_loss 0.00937
    # wandb:          val/mse_loss 0.00349
    # wandb:         val/topo_loss 0.15192
    # wandb:        val/total_loss 0.16478

