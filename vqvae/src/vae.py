import torch
import torch.nn as nn
import torch.nn.functional as F
from src.components import Encoder, Decoder
import math

class VAE(nn.Module):
    """
    Variational Autoencoder (VAE) using shared Encoder/Decoder.
    The encoder outputs features, which are flattened and mapped to latent distribution parameters (mu and logvar).
    A latent vector is sampled from this distribution, then mapped back to feature space for decoding.
    """
    def __init__(self, dim=128, latent_dim=512):
        super().__init__()
        self.enc = Encoder(dim)  # Encoder network
        self.dec = Decoder(dim)  # Decoder network
        self.dim = dim  # Feature dimension
        self.latent_dim = latent_dim  # Latent space dimension
        
        self.flattened_size = self.compute_flattened_size(torch.zeros(1, 3, 96, 96))
        
        self.linear_mu = nn.Linear(self.flattened_size, latent_dim)
        self.linear_logvar = nn.Linear(self.flattened_size, latent_dim)
        self.from_latent = nn.Linear(latent_dim, self.flattened_size)
        
    def compute_flattened_size(self, x):
        """
        Compute the size of the flattened feature map after encoding.
        Args:
            x (Tensor): Input tensor of shape (B, 3, H, W).
        Returns:
            int: Size of the flattened feature map per sample which is numel // batch_size.
            73728 = 128 * 24 * 24 for input size 96x96.
        """
        with torch.no_grad():
            encoded = self.enc(x)
            return encoded.numel() // encoded.size(0)

    def forward(self, x, beta=1.0):
        """
        Forward pass through the VAE.
        Args:
            x (Tensor): Input tensor of shape (B, 3, H, W).
            beta (float): Weight for the KL divergence loss term.
        Returns:
            reconstructed_x (Tensor): Reconstructed output tensor of shape (B, 3, H, W).
            total_loss (Tensor): Combined reconstruction and KL divergence loss.
        Instructions:
            1. Encode x to get feature map
            2. Extract mu and logvar from encoded features
            3. Sample latent vector using mu and logvar
            4. Map sampled latent vector back to feature space
            5. Decode
            6. Compute reconstruction loss (L1) and KL divergence loss
        """
        original_x = x # Shape: (B, 3, H, W)

        def inner_forward(x):
            """
            Inner forward pass to facilitate loss computation.
            Args:
                x (Tensor): Input tensor of shape (B, 3, H, W).
            Returns:
                reconstructed_x (Tensor): Reconstructed output tensor of shape (B, 3, H, W).
                mu (Tensor): Mean of the latent distribution, shape (B, latent_dim).
                logvar (Tensor): Log-variance of the latent distribution, shape (B, latent_dim).
            """
            # TODO: Implement steps 1-5 as described in the instructions. (Done)

            # 1. Encode x to get feature map
            encoded = torch.flatten(self.enc(x), start_dim=1)

            # 2. Extract mu and logvar from encoded features
            mu = self.linear_mu(encoded)
            logvar = self.linear_logvar(encoded)

            # 3. Sample latent vector using mu and logvar
            epsilon = torch.normal(torch.zeros(mu.shape), torch.ones(logvar.shape)).to(x.device)
            sampled = mu + torch.mul(epsilon, logvar.exp())

            # 4. Map sampled latent vector back to feature space
            decodeInput = self.from_latent(sampled)
            channels = self.dim
            flatShape = decodeInput.shape[1]
            image_dimensions = int(math.sqrt(flatShape / channels))
            decodeInput = decodeInput.reshape(decodeInput.shape[0], channels, image_dimensions, image_dimensions)

            # 5. Decode
            reconstructed_x = self.dec(decodeInput) # .reshape(x.shape)

            return reconstructed_x, mu, logvar

        # Step 6: Compute losses
        x, mu, logvar = inner_forward(x)
        reconstruction_loss = F.l1_loss(x, original_x, reduction='mean')
        kl_divergence = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        total_loss = reconstruction_loss + beta * kl_divergence
        return x, total_loss
