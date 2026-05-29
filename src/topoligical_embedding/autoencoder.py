import torch.nn as nn


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
