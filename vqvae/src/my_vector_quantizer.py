import torch
import torch.nn as nn
import torch.nn.functional as F

class MyVectorQuantizer(nn.Module):
    """
    A simplified custom Vector Quantizer (VQ) layer with EMA updates for stable training.
    
    This class quantizes input vectors by mapping them to the nearest codebook entries,
    using Exponential Moving Average (EMA) to update the codebook during training.
    It also includes optional dead code revival to prevent unused codes.
    
    Args:
        dim (int): Dimensionality of each vector (e.g., feature depth).
        codebook_size (int): Number of vectors in the codebook (vocabulary size).
        decay (float): EMA decay factor (higher = slower updates, default 0.99).
        commitment_weight (float): Weight for commitment loss (default 0.25).
        epsilon (float): Small value to avoid division by zero (default 1e-5).
        threshold_ema_dead_code (int): Threshold for considering a code "dead" (default 2).
    """
    def __init__(self, dim, codebook_size, decay=0.99, commitment_weight=0.25, epsilon=1e-5, threshold_ema_dead_code=2):
        super().__init__()
        self.dim = dim
        self.codebook_size = codebook_size
        self.decay = decay
        self.commitment_weight = commitment_weight
        self.epsilon = epsilon
        self.threshold_ema_dead_code = threshold_ema_dead_code
        
        # Codebook: Learnable embedding layer storing quantized vectors
        self.embedding = nn.Embedding(self.codebook_size, self.dim)
        
        # EMA buffers: Track cluster sizes and smoothed embeddings for stable updates
        self.register_buffer('ema_cluster_size', torch.zeros(codebook_size))
        self.register_buffer('ema_embeddings', self.embedding.weight.data.clone())
        self.register_buffer('initialized', torch.tensor(False, dtype=torch.bool))
    
    @torch.no_grad()
    def _reset_dead_codes(self, flattened_latents):
        """
        Replaces "dead" codes (those used below the threshold) by replacing them
        with random samples from the current batch. This prevents codebook collapse.
        """
        dead_mask = self.ema_cluster_size < self.threshold_ema_dead_code
        num_dead = dead_mask.sum().item()
        
        if num_dead > 0:
            # Sample replacement vectors from the batch
            batch_samples = flattened_latents[torch.randperm(flattened_latents.size(0))[:num_dead]]
            replacements = batch_samples.to(self.embedding.weight.dtype)
            
            # Update codebook and EMA buffers
            self.embedding.weight.data[dead_mask] = replacements
            self.ema_embeddings.data[dead_mask] = replacements
            self.ema_cluster_size[dead_mask] = 1.0

    def forward(self, encoded_latents):
        """
        Forward pass: Quantize input vectors and compute losses.
        
        Args:
            encoded_latents (Tensor): Encoder output, shape (B, N, D) where B=batch, N=tokens, D=dim.
        
        Returns:
            Tuple[Tensor, Tensor, Tensor]: (quantized vectors, indices, commitment loss).
        """
        # Step 1: Prepare inputs
        encoded_latents = encoded_latents.float()
        B, N, D = encoded_latents.shape # Shape: (B, N, D)
        flattened_latents = encoded_latents.reshape(-1, D)  # Flatten to (B*N, D) for vectorized ops
        
        def calculate_distances(z, e):
            """
            Compute squared Euclidean distances between each vector in z and the codebook vectors in e.
                using the formula: ||z - e||^2 = z^2 + e^2 - 2*z*e
            Args:
                z (Tensor): Input vectors, shape (B*N, D).
                e (Tensor): Codebook vectors, shape (codebook_size, D).
            Returns:
                Tensor: Distances, shape (B*N, codebook_size).
            """
            # TODO: Implement distance calculation as described in the instructions. (Done)

            # Check https://numpy.org/doc/stable/user/basics.broadcasting.html for broadcasting

            broad_z = torch.unsqueeze(z, 1) # (B*N, 1, D)
            # e = (codebook_size, D)

            res = torch.pow(broad_z, 2) + torch.pow(e, 2) - 2 * torch.mul(broad_z, e) # (B*N, codebook_size, D)
            return torch.sum(res, dim=-1) # (B*N, codebook_size)
        
        z = flattened_latents
        e = self.embedding.weight
        distances = calculate_distances(z, e)

        # Step 3: Find nearest codebook vectors
        def find_quantized_latents(distances_to_codebook):
            """
            Find the quantized latents and their indices from the distance matrix.
            Args:
                distances_to_codebook (Tensor): Distances, shape (B*N, codebook_size).
            Returns:
                Tuple[Tensor, Tensor]:
                    - quantized_latents_flat (Tensor): Quantized latents, shape (B*N, D).
                    - nearest_indices_flat (Tensor): Indices of nearest codebook entries, shape (B*N,).
            """
            # TODO: Implement finding nearest codebook vectors as described in the instructions. [Hint: Use argmin()] [One Line Each]
            # (Done)
            
            nearest_indices_flat = torch.argmin(distances_to_codebook, dim=1)
            quantized_latents_flat = torch.index_select(e, dim=0, index=nearest_indices_flat)

            return quantized_latents_flat, nearest_indices_flat

        quantized_latents_flat, indices_flat = find_quantized_latents(distances)
        quantized_latents = quantized_latents_flat.view(B, N, D)
        

        # Step 4: Update codebook via EMA (only during training)
        if self.training:
            # One-hot encode indices for aggregation
            encodings = F.one_hot(indices_flat, self.codebook_size).float()  # (B*N, codebook_size)
            cluster_counts = encodings.sum(0)  # (codebook_size,)
            weighted_sum_latents = torch.matmul(encodings.t(), flattened_latents)  # (codebook_size, D)
            
            # Update EMA buffers (restore the missing update for ema_cluster_size)
            self.ema_cluster_size.data.mul_(self.decay).add_(cluster_counts, alpha=1 - self.decay)
            self.ema_embeddings.data.mul_(self.decay).add_(weighted_sum_latents, alpha=1 - self.decay)
            
            # Normalize and copy to embedding (now use EMA cluster size for smoothing)
            # normalized_embeds = self.ema_embeddings / (self.ema_cluster_size.unsqueeze(1) + self.epsilon)
            def normalize_with_ema(ema_codebook, ema_cluster_sizes, epsilon):
                """
                Normalize the EMA codebook using the EMA cluster sizes to get updated embeddings.
                Args:
                    ema_codebook (Tensor): EMA codebook, shape (codebook_size, D). Think "Sum of vectors assigned to each code".
                    ema_cluster_sizes (Tensor): EMA cluster sizes, shape (codebook_size,). Think "Count of vectors assigned to each code".
                    epsilon (float): Small value to avoid division by zero.
                Returns:
                    Tensor: Normalized embeddings, shape (codebook_size, D). Think "Mean of vectors assigned to each code".
                """
                # TODO: Implement normalization as described in the instructions. [One Line] (Done)

                # I don't know if you're aware, but the one line is shown above in a comment
                return ema_codebook / (ema_cluster_sizes.unsqueeze(1) + epsilon)

            normalized_embeds = normalize_with_ema(self.ema_embeddings, self.ema_cluster_size, self.epsilon)
            self.embedding.weight.data.copy_(normalized_embeds)
            self._reset_dead_codes(flattened_latents)

        # Step 5: Compute commitment loss (encourages encoder to match quantized vectors)
        commitment_loss = self.commitment_weight * F.mse_loss(encoded_latents, quantized_latents.detach())
        
        # Step 6: Apply Straight-Through Estimator (STE) for gradient flow
        def apply_ste(quantized_latents, encoded_latents):
            """
            Apply the straight-through estimator to allow gradients to pass through
            the quantization operation unchanged during backpropagation.
            Args:
                quantized_latents (Tensor): Quantized output, shape (B, N, D).
                encoded_latents (Tensor): Original encoder output, shape (B, N, D).
            Returns:
                Tensor: Quantized latents with STE applied, shape (B, N, D).
            """
            # TODO: Implement STE as described in the instructions. [Hint: Use detach()] [One Line] 
            # (Done)
            return quantized_latents.detach() + encoded_latents - encoded_latents.detach()

        quantized_latents = apply_ste(quantized_latents, encoded_latents)
        quantized_indices = indices_flat.view(B, N)
        return quantized_latents, quantized_indices, commitment_loss
