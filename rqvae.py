import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm


class VectorQuantizer(nn.Module):
    """
    Vector Quantization module
    """
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super(VectorQuantizer, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        
        # Initialize embeddings
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.embeddings.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)
    
    def forward(self, inputs):
        # Convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        
        # Flatten input
        flat_input = inputs.view(-1, self.embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self.embeddings.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self.embeddings.weight.t()))
        
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self.embeddings.weight).view(input_shape)
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        # Straight through estimator
        quantized = inputs + (quantized - inputs).detach()
        
        # Convert quantized from BHWC -> BCHW
        return quantized.permute(0, 3, 1, 2).contiguous(), loss, encoding_indices.view(input_shape[:-1])


class ResidualVectorQuantizer(nn.Module):
    """
    Residual Quantizer that uses multiple VQ layers to quantize vectors
    """
    def __init__(self, num_embeddings, embedding_dim, num_quantizers=4):
        super(ResidualVectorQuantizer, self).__init__()
        self.num_quantizers = num_quantizers
        
        # Create multiple quantizers
        self.quantizers = nn.ModuleList([
            VectorQuantizer(num_embeddings, embedding_dim)
            for _ in range(num_quantizers)
        ])
    
    def forward(self, x):
        quantized = torch.zeros_like(x)
        residual = x
        
        total_loss = 0
        indices_list = []
        
        # Quantize residuals sequentially
        for i in range(self.num_quantizers):
            q, loss, indices = self.quantizers[i](residual)
            quantized = quantized + q
            residual = residual - q.detach()  # Detach to avoid backprop through this path
            total_loss += loss
            indices_list.append(indices)
            
        return quantized, total_loss, indices_list


# class LaplacianVQVAE(nn.Module):
class WaveletVQVAE(nn.Module):
    """
    VQ-VAE for Wavelet coefficients
    """
    def __init__(self, num_embeddings=1024, embedding_dim=64, num_quantizers=4, num_levels=3):
        super().__init__()
        self.num_levels = num_levels
        
        # Encoders for each level of the Laplacian pyramid
        self.encoders = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
                nn.ReLU(True),
                nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
                nn.ReLU(True),
                nn.Conv2d(128, embedding_dim, kernel_size=3, stride=1, padding=1)
            ) for _ in range(num_levels)
        ])
        
        # Residual Quantizers for each level
        self.quantizers = nn.ModuleList([
            VectorQuantizer(num_embeddings, embedding_dim)
            for _ in range(num_levels)
        ])
        
        # Decoders for each level
        self.decoders = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(embedding_dim, 128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(True),
                nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
                nn.ReLU(True),
                nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1)
            ) for _ in range(num_levels)
        ])
    
    def forward(self, pyramid_levels):
        """
        Forward pass through the VQ-VAE for each level of the Laplacian pyramid
        
        Args:
            pyramid_levels: List of tensors representing Laplacian pyramid levels
            
        Returns:
            reconstructed_levels: Reconstructed pyramid levels
            total_loss: Quantization loss
            indices_list: List of quantized indices for each level
        """
        assert len(pyramid_levels) == self.num_levels, "Number of pyramid levels doesn't match model"
        
        reconstructed_levels = []
        total_loss = 0
        all_indices = []
        
        for i in range(self.num_levels):
            # Encode
            encoded = self.encoders[i](pyramid_levels[i])
            
            # Quantize
            quantized, loss, indices = self.quantizers[i](encoded)
            total_loss += loss
            all_indices.append(indices)
            
            # Decode
            reconstructed = self.decoders[i](quantized)
            reconstructed_levels.append(reconstructed)
            
        return reconstructed_levels, total_loss, all_indices


# Utility functions for Laplacian pyramid
# Corrected function for building Laplacian pyramid
def build_laplacian_pyramid(image, num_levels=3):
    """
    Build a Laplacian pyramid from an image
    
    Args:
        image: Input tensor of shape [B, C, H, W]
        num_levels: Number of levels in the pyramid
        
    Returns:
        List of pyramid levels (tensors)
    """
    #print(f"Building Laplacian pyramid with {num_levels} levels")
    pyramids = []
    current = image
    
    for i in range(num_levels):
        # Downsample
        downsampled = F.interpolate(current, scale_factor=0.5, mode='bilinear', align_corners=False)
        
        # Upsample the downsampled image
        upsampled = F.interpolate(downsampled, size=current.shape[2:], mode='bilinear', align_corners=False)
        
        # Compute and store the residual (Laplacian)
        laplacian = current - upsampled
        pyramids.append(laplacian)
        
        # Update current for next level
        current = downsampled
    
    # Add the final downsampled image
    pyramids.append(current)
    
    return pyramids


def reconstruct_from_laplacian_pyramid(pyramid_levels):
    """
    Reconstruct an image from its Laplacian pyramid
    """
    current = pyramid_levels[-1]
    
    for i in range(len(pyramid_levels) - 2, -1, -1):
        # Upsample current
        upsampled = F.interpolate(current, size=pyramid_levels[i].shape[2:], 
                                 mode='bilinear', align_corners=False)
        
        # Add the residual
        current = upsampled + pyramid_levels[i]
    
    return current

class PositionalEncoding(nn.Module):
    """
    Positional encoding for the transformer models
    """
    def __init__(self, d_model, max_len=1000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x


class SpatialTransformer(nn.Module):
    """
    Transformer model for predicting spatial positions
    """
    def __init__(self, d_model=512, nhead=8, num_layers=6, 
                 dim_feedforward=2048, dropout=0.1, 
                 code_embedding_dim=64, num_embeddings=1024, max_len=1000):
        super(SpatialTransformer, self).__init__()
        
        # Embedding layer for code vectors
        self.code_embedding = nn.Linear(code_embedding_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        
        # Transformer layers
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, 
            dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # Final projection
        self.fc_out = nn.Linear(d_model, d_model)
        
    def forward(self, src, src_mask=None):
        """
        Args:
            src: Source sequence [batch, seq_len, code_embedding_dim]
            src_mask: Attention mask for autoregressive behavior
        """
        # Project to d_model
        src = self.code_embedding(src)
        
        # Add positional encoding
        src = self.pos_encoder(src)
        
        # Transpose for transformer: [seq_len, batch, d_model]
        src = src.transpose(0, 1)
        
        # Pass through transformer
        output = self.transformer_encoder(src, src_mask)
        
        # Transpose back: [batch, seq_len, d_model]
        output = output.transpose(0, 1)
        
        # Final projection
        output = self.fc_out(output)
        
        return output

class StandardTransformer(nn.Module):
    """
    Standard Transformer model for image generation using VQ-VAE tokens
    """
    def __init__(self, d_model=512, nhead=8, num_layers=6, 
                 dim_feedforward=2048, dropout=0.1, 
                 code_embedding_dim=64, num_embeddings=1024, max_len=1000):
        super(StandardTransformer, self).__init__()
        
        # Embedding layer for discrete codes
        self.code_embedding = nn.Embedding(num_embeddings, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        
        # Transformer layers
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, 
            dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # Output projection to vocabulary
        self.output_projection = nn.Linear(d_model, num_embeddings)
        
        # Model parameters
        self.d_model = d_model
        self.num_embeddings = num_embeddings
        
    def forward(self, src, target_indices=None, generation=False):
        """
        Forward pass through the transformer
        
        Args:
            src: Source tokens [batch, seq_len]
            target_indices: Target tokens for training [batch, seq_len] or None for inference
            generation: Whether we're in generation mode
        
        Returns:
            If generation=False: logits for next token prediction [batch, seq_len, vocab_size]
            If generation=True: predicted next token [batch, 1]
        """
        batch_size = src.shape[0]
        
        # Get embeddings
        src_embeddings = self.code_embedding(src)
        
        # Add positional encoding
        src_embeddings = self.pos_encoder(src_embeddings)
        
        # Create causal mask for autoregressive generation
        seq_len = src.shape[1]
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=src.device) * float('-inf'),
            diagonal=1
        )
        
        # Prepare for transformer: [seq_len, batch, d_model]
        src_embeddings = src_embeddings.transpose(0, 1)
        
        # Pass through transformer
        output = self.transformer_encoder(src_embeddings, mask=causal_mask)
        
        # Transpose back: [batch, seq_len, d_model]
        output = output.transpose(0, 1)
        
        # Project to vocabulary size
        logits = self.output_projection(output)
        
        if generation:
            # Return prediction for last position only (for autoregressive generation)
            next_token = torch.argmax(logits[:, -1:, :], dim=-1)
            return next_token
        else:
            # Return logits for all positions
            return logits
    

class DepthTransformer(nn.Module):
    """
    Transformer model for predicting depth tokens
    """
    def __init__(self, d_model=512, nhead=8, num_layers=2,
                 dim_feedforward=1024, dropout=0.1,
                 num_embeddings=1024, max_depth_positions=4):
        super(DepthTransformer, self).__init__()
        
        # Embedding layer for depth tokens
        self.token_embedding = nn.Embedding(num_embeddings, d_model)
        
        # Positional encoding for depth
        self.depth_pos_encoding = nn.Embedding(max_depth_positions, d_model)
        
        # Transformer layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        
        # Output layer
        self.fc_out = nn.Linear(d_model, num_embeddings)
        
    def forward(self, tgt, memory, tgt_mask=None):
        """
        Args:
            tgt: Target tokens [batch, depth_seq_len] or None at start
            memory: Memory from spatial transformer [batch, seq_len, d_model]
            tgt_mask: Mask for autoregressive generation
        """
        batch_size = memory.shape[0]
        depth_len = tgt.shape[1]
        
        # Create position indices
        positions = torch.arange(depth_len, device=tgt.device).unsqueeze(0).expand(batch_size, -1)
        
        # Get embeddings
        tgt_embeddings = self.token_embedding(tgt)
        pos_embeddings = self.depth_pos_encoding(positions)
        
        # Add positional encoding
        tgt_embeddings = tgt_embeddings + pos_embeddings
        
        # Prepare for transformer: [depth_seq_len, batch, d_model]
        tgt_embeddings = tgt_embeddings.transpose(0, 1)
        memory = memory.transpose(0, 1)
        
        # Transformer decoder
        output = self.transformer_decoder(tgt_embeddings, memory, tgt_mask=tgt_mask)
        
        # Back to [batch, depth_seq_len, d_model]
        output = output.transpose(0, 1)
        
        # Project to vocabulary
        logits = self.fc_out(output)
        
        return logits


class RQTransformer(nn.Module):
    """
    Combined model for Residual Quantized Transformer
    """
    def __init__(self, d_model=512, spatial_nhead=8, depth_nhead=4,
                 spatial_layers=6, depth_layers=2,
                 code_embedding_dim=64, num_embeddings=1024, 
                 num_quantizers=4, max_seq_len=64):
        super(RQTransformer, self).__init__()
        
        self.num_embeddings = num_embeddings
        self.num_quantizers = num_quantizers
        
        # Spatial transformer for modeling spatial dependencies
        self.spatial_transformer = SpatialTransformer(
            d_model=d_model,
            nhead=spatial_nhead,
            num_layers=spatial_layers,
            code_embedding_dim=code_embedding_dim,
            num_embeddings=num_embeddings,
            max_len=max_seq_len
        )
        
        # Depth transformer for predicting tokens at different depths
        self.depth_transformer = DepthTransformer(
            d_model=d_model,
            nhead=depth_nhead,
            num_layers=depth_layers,
            num_embeddings=num_embeddings,
            num_quantizers=num_quantizers
        )
        
        # Start-of-sequence token embedding
        self.sos_embedding = nn.Parameter(torch.randn(1, 1, code_embedding_dim))
        
    def forward(self, code_embeddings, target_indices=None, generation=False):
        """
        Forward pass
        
        Args:
            code_embeddings: Embeddings of codes from previous positions [batch, seq_len, code_embedding_dim]
            target_indices: Target indices for training [batch, seq_len, depth] or None for inference
            generation: Whether we're generating (inference) or training
        """
        batch_size = code_embeddings.shape[0]
        
        # Add SOS token at the beginning
        code_embeddings = torch.cat([
            self.sos_embedding.expand(batch_size, -1, -1),
            code_embeddings
        ], dim=1)
        
        # Create causal mask for spatial transformer
        seq_len = code_embeddings.shape[1]
        spatial_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=code_embeddings.device) * float('-inf'),
            diagonal=1
        )
        
        # Run through spatial transformer
        spatial_output = self.spatial_transformer(code_embeddings, spatial_mask)
        
        if generation:
            # For inference, return just the last position's output
            last_position = spatial_output[:, -1:, :]
            
            # Initialize with start tokens
            current_depth_tokens = torch.zeros(
                (batch_size, 1), dtype=torch.long, device=code_embeddings.device
            )
            
            all_depth_tokens = []
            
            # Generate one token at a time for each depth
            for d in range(self.num_quantizers):
                # Create causal mask for depth
                depth_len = current_depth_tokens.shape[1]
                depth_mask = torch.triu(
                    torch.ones(depth_len, depth_len, device=code_embeddings.device) * float('-inf'),
                    diagonal=1
                )
                
                # Predict next token
                depth_logits = self.depth_transformer(
                    current_depth_tokens, last_position, depth_mask
                )
                
                # Get most likely token
                next_token = torch.argmax(depth_logits[:, -1, :], dim=-1, keepdim=True)
                
                # Add to sequence
                current_depth_tokens = torch.cat([current_depth_tokens, next_token], dim=1)
                all_depth_tokens.append(next_token)
            
            # Return the predicted depth tokens (excluding start token)
            return torch.cat(all_depth_tokens, dim=1)
        
        else:
            # For training, predict all positions except the last one (which has no target)
            outputs = []
            
            for i in range(1, seq_len):
                current_position = spatial_output[:, i:i+1, :]
                
                # Get target indices for this position
                target_for_position = target_indices[:, i-1, :]
                
                # Initialize with start token
                current_depth_tokens = torch.zeros(
                    (batch_size, 1), dtype=torch.long, device=code_embeddings.device
                )
                
                # Teacher forcing - use target tokens as input
                for d in range(self.num_quantizers):
                    # Add target token from previous depth
                    if d > 0:
                        current_depth_tokens = torch.cat([
                            current_depth_tokens, 
                            target_for_position[:, d-1:d]
                        ], dim=1)
                    
                    # Create mask for depth
                    depth_len = current_depth_tokens.shape[1]
                    depth_mask = torch.triu(
                        torch.ones(depth_len, depth_len, device=code_embeddings.device) * float('-inf'),
                        diagonal=1
                    )
                    
                    # Predict next token
                    depth_logits = self.depth_transformer(
                        current_depth_tokens, current_position, depth_mask
                    )
                    
                    # Get prediction for this depth
                    pred = depth_logits[:, -1:, :]
                    outputs.append(pred)
            
            # Stack all predictions
            return torch.stack(outputs, dim=1)


# Training function
def train_laplacian_vqvae(model, dataloader, optimizer, epochs=10, device='cuda'):
    """
    Train the Laplacian VQ-VAE model
    """
    model.to(device)
    model.train()
    print("Training Laplacian VQ-VAE...")
    for epoch in range(epochs):
        total_loss = 0
        loop = tqdm(dataloader, desc=f"Epoch {epoch}", leave=False)

        for batch_idx, (data, _) in enumerate(loop):   # Unpack properly - datasets return (data, label)
            data = data.to(device)
            
            # Build Laplacian pyramid
            laplacian_levels = build_laplacian_pyramid(data, num_levels=model.num_levels-1)
            
            # Forward pass
            reconstructed_levels, vq_loss, indices, _ = model(laplacian_levels)
            
            # Compute reconstruction loss
            recon_loss = 0
            for i in range(len(laplacian_levels)):
                recon_loss += F.mse_loss(reconstructed_levels[i], laplacian_levels[i])
            
            # Total loss
            loss = recon_loss + vq_loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

            loop.set_postfix(loss=loss.item())
            
            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}')
        
        print(f'Epoch: {epoch}, Average Loss: {total_loss/len(dataloader):.4f}')
        
    return model

def train_standard_transformer(model, vqvae_model, dataloader, optimizer, epochs=10, device='cuda'):
    """
    Train the standard transformer model
    """
    model.to(device)
    vqvae_model.to(device)
    model.train()
    vqvae_model.eval()  # Freeze VQ-VAE
    
    criterion = nn.CrossEntropyLoss()
    
    for epoch in tqdm(range(epochs)):
        total_loss = 0
        
        for batch_idx, data in enumerate(dataloader):
            data = data.to(device)
            batch_size = data.shape[0]
            
            # Get tokenized representation from VQ-VAE
            with torch.no_grad():
                laplacian_levels = build_laplacian_pyramid(data, num_levels=vqvae_model.num_levels-1)
                _, _, all_indices, _ = vqvae_model(laplacian_levels, training=False)
            
            # Flatten and combine indices for all levels and depths
            # For simplicity, we'll concatenate all tokens into a single sequence
            flattened_tokens = []
            for level_indices in all_indices:
                for depth_indices in level_indices:
                    # Flatten spatial dimensions
                    flat_tokens = depth_indices.reshape(batch_size, -1)
                    flattened_tokens.append(flat_tokens)
            
            # Concatenate all tokens into one sequence per batch
            # Shape: [batch_size, seq_len]
            token_sequences = torch.cat(flattened_tokens, dim=1)
            
            # Create input and target sequences for next-token prediction
            # Input: tokens[:-1], Target: tokens[1:]
            src_tokens = token_sequences[:, :-1]
            tgt_tokens = token_sequences[:, 1:]
            
            # Forward pass
            logits = model(src_tokens, tgt_tokens)
            
            # Compute loss
            # Reshape for cross entropy: [batch_size * seq_len, vocab_size]
            logits_flat = logits.reshape(-1, model.num_embeddings)
            targets_flat = tgt_tokens.reshape(-1)
            
            loss = criterion(logits_flat, targets_flat)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 50 == 0:
                print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}')
        
        print(f'Epoch: {epoch}, Average Loss: {total_loss/len(dataloader):.4f}')
        
    return model

# Training function for RQ Transformer
def train_rq_transformer(model, vqvae_model, dataloader, optimizer, epochs=10, device='cuda'):
    """
    Train the RQ Transformer model
    """
    model.to(device)
    vqvae_model.to(device)
    model.train()
    vqvae_model.eval()  # Freeze VQ-VAE
    
    criterion = nn.CrossEntropyLoss()
    
    for epoch in tqdm(range(epochs)):
        total_loss = 0
        
        for batch_idx, data in enumerate(dataloader):
            data = data.to(device)
            batch_size = data.shape[0]
            
            # Get tokenized representation from VQ-VAE
            with torch.no_grad():
                laplacian_levels = build_laplacian_pyramid(data, num_levels=vqvae_model.num_levels-1)
                _, _, all_indices = vqvae_model(laplacian_levels)
            
            # Rearrange indices for autoregressive prediction
            # For simplicity, let's assume we're working with flattened images
            indices_flattened = []
            for level_indices in all_indices:
                for depth_indices in level_indices:
                    # Flatten spatial dimensions
                    flat_indices = depth_indices.view(batch_size, -1)
                    indices_flattened.append(flat_indices)
            
            # Stack depth-wise
            indices_stacked = torch.stack(indices_flattened, dim=-1)
            
            # Create input embeddings (excluding the last position)
            code_embeddings = []
            for i in range(indices_stacked.shape[1] - 1):
                pos_indices = indices_stacked[:, i, :]
                # Get embeddings for all depths at this position
                embed = torch.zeros(batch_size, vqvae_model.quantizers[0].quantizers[0].embedding_dim, device=device)
                for d, idx in enumerate(pos_indices.unbind(-1)):
                    embed += vqvae_model.quantizers[0].quantizers[d].embeddings(idx)
                code_embeddings.append(embed.unsqueeze(1))
            
            # Stack all positions
            if code_embeddings:
                code_embeddings = torch.cat(code_embeddings, dim=1)
                
                # Target is the next position's indices
                target_indices = indices_stacked[:, 1:, :]
                
                # Forward pass
                outputs = model(code_embeddings, target_indices)
                
                # Compute loss
                loss = 0
                for i, output in enumerate(outputs):
                    # Determine which target we're predicting
                    pos_idx = i // model.num_quantizers
                    depth_idx = i % model.num_quantizers
                    target = target_indices[:, pos_idx, depth_idx]
                    
                    # Cross entropy loss
                    loss += criterion(output.view(-1, model.num_embeddings), target.view(-1))
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if batch_idx % 50 == 0:
                print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}')
        
        print(f'Epoch: {epoch}, Average Loss: {total_loss/len(dataloader):.4f}')
        
    return model


# Example of how to generate images using the trained models
def generate_images(rq_transformer, vqvae_model, num_images=1, device='cuda'):
    """
    Generate images using the trained models
    """
    rq_transformer.eval()
    vqvae_model.eval()
    
    with torch.no_grad():
        # Start with empty embeddings
        batch_size = num_images
        seq_len = 8 * 8  # For 256x256 images with 32x32 tokens
        code_embedding_dim = vqvae_model.quantizers[0].quantizers[0].embedding_dim
        
        # Empty embeddings
        code_embeddings = torch.zeros(batch_size, 0, code_embedding_dim, device=device)
        
        # Generate tokens autoregressively
        all_tokens = []
        
        for i in range(seq_len):
            # Predict next position
            next_tokens = rq_transformer(code_embeddings, generation=True)
            all_tokens.append(next_tokens)
            
            # Get embeddings for the predicted tokens
            next_embedding = torch.zeros(batch_size, code_embedding_dim, device=device)
            for d, tokens in enumerate(next_tokens.unbind(-1)):
                next_embedding += vqvae_model.quantizers[0].quantizers[d].embeddings(tokens)
            
            # Add to code embeddings
            code_embeddings = torch.cat([
                code_embeddings, 
                next_embedding.unsqueeze(1)
            ], dim=1)
        
        # Convert tokens back to images
        # This is a simplified version - you'd need to reshape and use the proper decoders
        all_tokens = torch.stack(all_tokens, dim=1)  # [batch, seq_len, depth]
        
        # Reshape back to spatial dimensions
        h, w = 8, 8  # Assuming 8x8 tokens for a 256x256 image
        reshaped_tokens = all_tokens.view(batch_size, h, w, -1)
        
        # Decode each level
        reconstructed_levels = []
        for level in range(vqvae_model.num_levels):
            level_tokens = reshaped_tokens[:, :, :, level * vqvae_model.quantizers[0].num_quantizers: 
                                             (level + 1) * vqvae_model.quantizers[0].num_quantizers]
            
            # Convert tokens to quantized vectors
            quantized = torch.zeros(batch_size, code_embedding_dim, h, w, device=device)
            for d in range(vqvae_model.quantizers[0].num_quantizers):
                tokens = level_tokens[:, :, :, d].view(batch_size, -1)
                embeddings = vqvae_model.quantizers[level].quantizers[d].embeddings(tokens)
                embeddings = embeddings.view(batch_size, h, w, -1).permute(0, 3, 1, 2)
                quantized += embeddings
            
            # Decode
            reconstructed = vqvae_model.decoders[level](quantized)
            reconstructed_levels.append(reconstructed)
        
        # Reconstruct from Laplacian pyramid
        images = reconstruct_from_laplacian_pyramid(reconstructed_levels)
        
        return images
    
def generate_images_with_standard_transformer(transformer, vqvae_model, num_images=1, device='cuda'):
    """
    Generate images using the standard transformer and VQ-VAE
    """
    transformer.eval()
    vqvae_model.eval()
    
    with torch.no_grad():
        batch_size = num_images
        
        # Calculate sequence length from VQ-VAE architecture
        # This is the total number of tokens needed for a complete image
        # Assuming all levels and depths are concatenated
        token_len_per_level = []
        for level in range(vqvae_model.num_levels):
            # Estimate spatial dimensions after encoding
            # This is a simplified calculation and may need adjustment
            h = w = 256 // (2**(level+2))  # Assuming 256x256 input and downsampling
            tokens_in_level = h * w * vqvae_model.num_quantizers
            token_len_per_level.append(tokens_in_level)
        
        total_seq_len = sum(token_len_per_level)
        
        # Start with a single start token (0)
        generated_sequence = torch.zeros((batch_size, 1), dtype=torch.long, device=device)
        
        # Generate tokens autoregressively
        for i in range(total_seq_len):
            # Predict next token
            next_token = transformer(generated_sequence, generation=True)
            
            # Add to sequence
            generated_sequence = torch.cat([generated_sequence, next_token], dim=1)
            
            if i % 100 == 0:
                print(f"Generated {i}/{total_seq_len} tokens")
        
        # Remove the initial start token
        generated_sequence = generated_sequence[:, 1:]
        
        # Split the sequence back into levels and depths
        split_indices = []
        current_idx = 0
        for level_len in token_len_per_level:
            split_indices.append((current_idx, current_idx + level_len))
            current_idx += level_len
        
        # Reconstruct the Laplacian pyramid
        reconstructed_levels = []
        
        for level in range(vqvae_model.num_levels):
            start_idx, end_idx = split_indices[level]
            level_tokens = generated_sequence[:, start_idx:end_idx]
            
            # Reshape back to spatial dimensions
            h = w = 256 // (2**(level+2))  # Same calculation as above
            level_tokens = level_tokens.reshape(batch_size, h, w, vqvae_model.num_quantizers)
            
            # Convert tokens to quantized vectors
            quantized = torch.zeros(batch_size, vqvae_model.quantizers[0].embedding_dim, h, w, device=device)
            
            for d in range(vqvae_model.num_quantizers):
                depth_tokens = level_tokens[:, :, :, d].reshape(batch_size, -1)
                # Get embeddings for these tokens
                if hasattr(vqvae_model.quantizers[level], 'codebook'):
                    # For ImprovedLaplacianVQVAE
                    embeddings = vqvae_model.quantizers[level].codebook(depth_tokens)
                else:
                    # For LaplacianVQVAE
                    embeddings = vqvae_model.quantizers[level].quantizers[d].embeddings(depth_tokens)
                
                embeddings = embeddings.reshape(batch_size, h, w, -1).permute(0, 3, 1, 2)
                quantized += embeddings
            
            # Decode this level
            reconstructed = vqvae_model.decoders[level](quantized)
            reconstructed_levels.append(reconstructed)
        
        # Reconstruct full image from Laplacian pyramid
        generated_images = reconstruct_from_laplacian_pyramid(reconstructed_levels)
        
        return generated_images
    
# Additional functionality for model training and generation

# Stochastic sampling for residual quantizer
class StochasticResidualQuantizer(nn.Module):
    """
    Stochastic Residual Quantizer that implements soft labeling and sampling
    """
    def __init__(self, num_embeddings, embedding_dim, num_quantizers=4, temperature=1.0):
        super(StochasticResidualQuantizer, self).__init__()
        self.num_quantizers = num_quantizers
        self.temperature = temperature
        
        # Create multiple quantizers with shared codebook
        self.codebook = nn.Embedding(num_embeddings, embedding_dim)
        self.codebook.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)
    
    def forward(self, x, training=True):
        quantized = torch.zeros_like(x)
        residual = x
        
        total_loss = 0
        indices_list = []
        soft_targets_list = []
        
        # Quantize residuals sequentially
        for i in range(self.num_quantizers):
            # Convert inputs from BCHW -> BHWC
            residual_i = residual.permute(0, 2, 3, 1).contiguous()
            input_shape = residual_i.shape
            
            # Flatten input
            flat_residual = residual_i.view(-1, residual_i.shape[-1])
            
            # Calculate distances
            distances = (torch.sum(flat_residual**2, dim=1, keepdim=True) 
                        + torch.sum(self.codebook.weight**2, dim=1)
                        - 2 * torch.matmul(flat_residual, self.codebook.weight.t()))
            
            # Convert to distribution with temperature
            logits = -distances / self.temperature
            probs = F.softmax(logits, dim=1)
            
            # Store soft targets for training
            soft_targets = probs.view(*input_shape[:-1], -1)
            soft_targets_list.append(soft_targets)
            
            if training:
                # Stochastic sampling during training
                dist = torch.distributions.Categorical(probs)
                encoding_indices = dist.sample().unsqueeze(1)
            else:
                # Deterministic sampling during inference
                encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
            
            # One-hot encoding
            encodings = torch.zeros(encoding_indices.shape[0], self.codebook.num_embeddings, 
                                   device=residual.device)
            encodings.scatter_(1, encoding_indices, 1)
            
            # Quantize
            quantized_residual = torch.matmul(encodings, self.codebook.weight).view(input_shape)
            
            # Add to quantized representation
            quantized_residual = quantized_residual.permute(0, 3, 1, 2).contiguous()
            quantized = quantized + quantized_residual
            
            # Update residual - detach for straight-through estimator
            residual = residual - quantized_residual.detach()
            
            # Store indices
            indices = encoding_indices.view(input_shape[:-1])
            indices_list.append(indices)
            
            # Commitment loss
            e_latent_loss = F.mse_loss(quantized_residual.detach(), residual_i.permute(0, 3, 1, 2))
            q_latent_loss = F.mse_loss(quantized_residual, residual_i.permute(0, 3, 1, 2).detach())
            total_loss += q_latent_loss + 0.25 * e_latent_loss
            
        return quantized, total_loss, indices_list, soft_targets_list


# Improved LaplacianVQVAE with stochastic sampling
class ImprovedLaplacianVQVAE(nn.Module):
    """
    Improved VQ-VAE for Laplacian pyramid levels with stochastic sampling and soft labeling
    """
    def __init__(self, num_embeddings=1024, embedding_dim=64, num_quantizers=4, num_levels=3, temperature=1.0):
        super(ImprovedLaplacianVQVAE, self).__init__()
        self.num_levels = num_levels
        self.num_quantizers = num_quantizers
        
        # Encoders for each level of the Laplacian pyramid
        self.encoders = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(256, embedding_dim, kernel_size=3, stride=1, padding=1)
            ) for _ in range(num_levels)
        ])
        
        # Stochastic Residual Quantizers for each level
        self.quantizers = nn.ModuleList([
            StochasticResidualQuantizer(
                num_embeddings, embedding_dim, num_quantizers, temperature
            ) for _ in range(num_levels)
        ])
        
        # Decoders for each level
        self.decoders = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(embedding_dim, 256, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
                nn.Tanh()
            ) for _ in range(num_levels)
        ])
    
    def forward(self, pyramid_levels, training=True):
        """
        Forward pass through the VQ-VAE for each level of the Laplacian pyramid
        
        Args:
            pyramid_levels: List of tensors representing Laplacian pyramid levels
            training: Whether this is training or inference mode
            
        Returns:
            reconstructed_levels: Reconstructed pyramid levels
            total_loss: Quantization loss
            indices_list: List of quantized indices for each level
            soft_targets_list: List of soft targets for each level (for soft labeling)
        """
        assert len(pyramid_levels) == self.num_levels, "Number of pyramid levels doesn't match model"
        
        reconstructed_levels = []
        total_loss = 0
        all_indices = []
        all_soft_targets = []
        
        for i in range(self.num_levels):
            # Encode
            encoded = self.encoders[i](pyramid_levels[i])
            
            # Quantize with stochastic sampling
            quantized, loss, indices, soft_targets = self.quantizers[i](encoded, training=training)
            total_loss += loss
            all_indices.append(indices)
            all_soft_targets.append(soft_targets)
            
            # Decode
            reconstructed = self.decoders[i](quantized)
            reconstructed_levels.append(reconstructed)
            
        return reconstructed_levels, total_loss, all_indices, all_soft_targets


# Improved RQ-Transformer with soft labeling
class ImprovedRQTransformer(nn.Module):
    """
    Combined model for Residual Quantized Transformer with soft labeling
    """
    def __init__(self, d_model=512, spatial_nhead=8, depth_nhead=4,
                 spatial_layers=6, depth_layers=2,
                 code_embedding_dim=64, num_embeddings=1024, 
                 num_quantizers=4, max_seq_len=64):
        super(ImprovedRQTransformer, self).__init__()
        
        self.num_embeddings = num_embeddings
        self.num_quantizers = num_quantizers
        self.d_model = d_model
        
        # Spatial transformer for modeling spatial dependencies
        self.spatial_transformer = SpatialTransformer(
            d_model=d_model,
            nhead=spatial_nhead,
            num_layers=spatial_layers,
            code_embedding_dim=code_embedding_dim,
            num_embeddings=num_embeddings,
            max_len=max_seq_len
        )
        
        # Depth transformer for predicting tokens at different depths
        self.depth_transformer = DepthTransformer(
            d_model=d_model,
            nhead=depth_nhead,
            num_layers=depth_layers,
            num_embeddings=num_embeddings,
            num_quantizers=num_quantizers
        )
        
        # Start-of-sequence token embedding
        self.sos_embedding = nn.Parameter(torch.randn(1, 1, code_embedding_dim))
        
    def forward(self, code_embeddings, target_indices=None, soft_targets=None, generation=False):
        """
        Forward pass
        
        Args:
            code_embeddings: Embeddings of codes from previous positions [batch, seq_len, code_embedding_dim]
            target_indices: Target indices for training [batch, seq_len, depth] or None for inference
            soft_targets: Soft targets for training [batch, seq_len, depth, num_embeddings] or None
            generation: Whether we're generating (inference) or training
        """
        batch_size = code_embeddings.shape[0]
        
        # Add SOS token at the beginning
        code_embeddings = torch.cat([
            self.sos_embedding.expand(batch_size, -1, -1),
            code_embeddings
        ], dim=1)
        
        # Create causal mask for spatial transformer
        seq_len = code_embeddings.shape[1]
        spatial_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=code_embeddings.device) * float('-inf'),
            diagonal=1
        )
        
        # Run through spatial transformer
        spatial_output = self.spatial_transformer(code_embeddings, spatial_mask)
        
        if generation:
            # For inference, return just the last position's output
            last_position = spatial_output[:, -1:, :]
            
            # Initialize with start tokens
            current_depth_tokens = torch.zeros(
                (batch_size, 1), dtype=torch.long, device=code_embeddings.device
            )
            
            all_depth_tokens = []
            
            # Generate one token at a time for each depth
            for d in range(self.num_quantizers):
                # Create causal mask for depth
                depth_len = current_depth_tokens.shape[1]
                depth_mask = torch.triu(
                    torch.ones(depth_len, depth_len, device=code_embeddings.device) * float('-inf'),
                    diagonal=1
                )
                
                # Predict next token
                depth_logits = self.depth_transformer(
                    current_depth_tokens, last_position, depth_mask
                )
                
                # Get most likely token
                next_token = torch.argmax(depth_logits[:, -1, :], dim=-1, keepdim=True)
                
                # Add to sequence
                current_depth_tokens = torch.cat([current_depth_tokens, next_token], dim=1)
                all_depth_tokens.append(next_token)
            
            # Return the predicted depth tokens (excluding start token)
            return torch.cat(all_depth_tokens, dim=1)
        
        else:
            # For training, predict all positions except the last one
            all_logits = []
            
            for i in range(1, seq_len):
                current_position = spatial_output[:, i:i+1, :]
                
                # Get target indices for this position
                target_for_position = target_indices[:, i-1, :]
                
                if soft_targets is not None:
                    soft_target_for_position = soft_targets[:, i-1, :, :]
                
                # Initialize with start token
                current_depth_tokens = torch.zeros(
                    (batch_size, 1), dtype=torch.long, device=code_embeddings.device
                )
                
                # Teacher forcing - use target tokens as input
                position_logits = []
                
                for d in range(self.num_quantizers):
                    # Add target token from previous depth
                    if d > 0:
                        current_depth_tokens = torch.cat([
                            current_depth_tokens, 
                            target_for_position[:, d-1:d]
                        ], dim=1)
                    
                    # Create mask for depth
                    depth_len = current_depth_tokens.shape[1]
                    depth_mask = torch.triu(
                        torch.ones(depth_len, depth_len, device=code_embeddings.device) * float('-inf'),
                        diagonal=1
                    )
                    
                    # Predict next token
                    depth_logits = self.depth_transformer(
                        current_depth_tokens, current_position, depth_mask
                    )
                    
                    # Get prediction for this depth
                    pred = depth_logits[:, -1, :]
                    position_logits.append(pred)
                
                # Add all logits for this position
                all_logits.append(torch.stack(position_logits, dim=1))
            
            # Stack all predictions [batch, seq_len-1, depth, num_embeddings]
            return torch.stack(all_logits, dim=1)


# Improved training function with soft labeling
def train_improved_rq_transformer(model, vqvae_model, dataloader, optimizer, 
                                epochs=10, device='cuda'):
    """
    Train the improved RQ Transformer model with soft labeling
    """
    model.to(device)
    vqvae_model.to(device)
    model.train()
    vqvae_model.eval()  # Freeze VQ-VAE
    
    kl_loss_fn = nn.KLDivLoss(reduction='batchmean')
    
    for epoch in tqdm(range(epochs)):
        total_loss = 0
        
        for batch_idx, data in enumerate(dataloader):
            data = data.to(device)
            batch_size = data.shape[0]
            
            # Get tokenized representation from VQ-VAE
            with torch.no_grad():
                laplacian_levels = build_laplacian_pyramid(data, num_levels=vqvae_model.num_levels-1)
                _, _, all_indices, all_soft_targets = vqvae_model(laplacian_levels, training=False)
            
            # Process indices and soft targets for training
            indices_flattened = []
            soft_targets_flattened = []
            
            for level in range(len(all_indices)):
                for depth in range(len(all_indices[level])):
                    # Flatten spatial dimensions
                    indices = all_indices[level][depth]
                    flat_indices = indices.view(batch_size, -1)
                    indices_flattened.append(flat_indices)
                    
                    # Flatten soft targets
                    soft_target = all_soft_targets[level][depth]
                    flat_soft_target = soft_target.view(batch_size, -1, model.num_embeddings)
                    soft_targets_flattened.append(flat_soft_target)
            
            # Stack by level and depth
            indices_stacked = torch.stack(indices_flattened, dim=-1)  # [batch, spatial, level*depth]
            soft_targets_stacked = torch.stack(soft_targets_flattened, dim=1)  # [batch, level*depth, spatial, num_emb]
            
            # Create input embeddings (excluding the last position)
            code_embeddings = []
            for i in range(indices_stacked.shape[1] - 1):
                pos_indices = indices_stacked[:, i, :]
                # Get embeddings for all depths at this position
                embed = torch.zeros(batch_size, vqvae_model.quantizers[0].embedding_dim, device=device)
                for d, idx in enumerate(pos_indices.unbind(-1)):
                    embed += vqvae_model.quantizers[0].codebook(idx)
                code_embeddings.append(embed.unsqueeze(1))
            
            # Stack positions
            if code_embeddings:
                code_embeddings = torch.cat(code_embeddings, dim=1)
                
                # Target indices
                target_indices = indices_stacked[:, 1:, :]
                
                # Soft targets (reorganize for transformer output)
                # From [batch, level*depth, spatial, num_emb] to [batch, spatial-1, depth, num_emb]
                soft_targets = soft_targets_stacked[:, :, 1:, :]
                soft_targets = soft_targets.permute(0, 2, 1, 3)
                
                # Forward pass
                logits = model(code_embeddings, target_indices, soft_targets)
                
                # Compute loss using soft labels (KL divergence)
                loss = 0
                # logits shape: [batch, seq_len-1, depth, num_embeddings]
                
                for i in range(logits.shape[1]):  # Over positions
                    for d in range(logits.shape[2]):  # Over depths
                        pred_logits = logits[:, i, d]
                        target_dist = soft_targets[:, i, d]
                        
                        # KL divergence between predicted distribution and target distribution
                        # Convert logits to log probabilities
                        log_probs = F.log_softmax(pred_logits, dim=-1)
                        loss += kl_loss_fn(log_probs, target_dist)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if batch_idx % 50 == 0:
                print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}')
        
        print(f'Epoch: {epoch}, Average Loss: {total_loss/len(dataloader):.4f}')
        
    return model


# Visualization utilities
def visualize_reconstruction(model, data, device='cuda'):
    """
    Visualize original and reconstructed images
    """
    model.eval()
    with torch.no_grad():
        data = data.to(device)
        
        # Build Laplacian pyramid
        laplacian_levels = build_laplacian_pyramid(data, num_levels=model.num_levels-1)
        
        # Reconstruct
        reconstructed_levels, _, _, _ = model(laplacian_levels, training=False)
        
        # Rebuild image from Laplacian pyramid
        reconstructed = reconstruct_from_laplacian_pyramid(reconstructed_levels)
        
        # Denormalize
        data = (data + 1) / 2.0
        reconstructed = (reconstructed + 1) / 2.0
        
        # Move to CPU for visualization
        data = data.cpu()
        reconstructed = reconstructed.cpu()
        
        # Plot
        num_images = min(4, data.shape[0])
        fig, axes = plt.subplots(2, num_images, figsize=(num_images * 3, 6))
        
        for i in range(num_images):
            # Original
            axes[0, i].imshow(data[i].permute(1, 2, 0))
            axes[0, i].set_title('Original')
            axes[0, i].axis('off')
            
            # Reconstructed
            axes[1, i].imshow(reconstructed[i].permute(1, 2, 0))
            axes[1, i].set_title('Reconstructed')
            axes[1, i].axis('off')
        
        plt.tight_layout()
        return fig


def visualize_laplacian_pyramid(pyramid_levels):
    """
    Visualize the levels of a Laplacian pyramid
    """
    num_levels = len(pyramid_levels)
    
    # Denormalize
    pyramid_levels = [(level + 1) / 2.0 for level in pyramid_levels]
    
    # Move to CPU
    pyramid_levels = [level.cpu() for level in pyramid_levels]
    
    # Plot
    fig, axes = plt.subplots(1, num_levels, figsize=(num_levels * 3, 3))
    
    for i, level in enumerate(pyramid_levels):
        if num_levels == 1:
            ax = axes
        else:
            ax = axes[i]
        
        # Get first image
        img = level[0].permute(1, 2, 0).numpy()
        
        # Clip to valid range
        img = np.clip(img, 0, 1)
        
        # Plot
        ax.imshow(img)
        ax.set_title(f'Level {i}')
        ax.axis('off')
    
    plt.tight_layout()
    return fig


def interpolate_in_latent_space(model, data1, data2, steps=10, device='cuda'):
    """
    Generate interpolations between two images in the latent space
    """
    model.eval()
    with torch.no_grad():
        data1 = data1.unsqueeze(0).to(device)
        data2 = data2.unsqueeze(0).to(device)
        
        # Get encoded representations
        laplacian1 = build_laplacian_pyramid(data1, num_levels=model.num_levels-1)
        laplacian2 = build_laplacian_pyramid(data2, num_levels=model.num_levels-1)
        
        encoded1 = [model.encoders[i](laplacian1[i]) for i in range(len(laplacian1))]
        encoded2 = [model.encoders[i](laplacian2[i]) for i in range(len(laplacian2))]
        
        # Interpolate
        interpolations = []
        for alpha in np.linspace(0, 1, steps):
            # Interpolate each level
            interpolated_levels = []
            for i in range(len(encoded1)):
                # Linear interpolation
                interpolated = alpha * encoded1[i] + (1 - alpha) * encoded2[i]
                
                # Quantize
                quantized, _, _, _ = model.quantizers[i](interpolated, training=False)
                
                # Decode
                decoded = model.decoders[i](quantized)
                interpolated_levels.append(decoded)
            
            # Reconstruct image
            interpolated_image = reconstruct_from_laplacian_pyramid(interpolated_levels)
            interpolations.append(interpolated_image)
        
        # Stack results
        interpolations = torch.cat(interpolations, dim=0)
        
        # Denormalize
        interpolations = (interpolations + 1) / 2.0
        
        return interpolations.cpu()


# Main training script with improved models
# if __name__ == "__main__":
#     import torch
#     import torchvision
#     from torchvision import datasets, transforms
#     import matplotlib.pyplot as plt
    
#     # Hyperparameters
#     batch_size = 32
#     vqvae_epochs = 15
#     transformer_epochs = 10
#     learning_rate = 3e-4
#     num_embeddings = 1024
#     embedding_dim = 64
#     num_quantizers = 4
#     num_levels = 3
#     temperature = 0.5  # Temperature for soft sampling
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
#     # Dataset
#     transform = transforms.Compose([
#         transforms.Resize(256),
#         transforms.CenterCrop(256),
#         transforms.ToTensor(),
#         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#     ])
    
#     # Use any dataset
#     dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
#     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
#     # Create improved models
#     vqvae_model = ImprovedLaplacianVQVAE(
#         num_embeddings=num_embeddings,
#         embedding_dim=embedding_dim,
#         num_quantizers=num_quantizers,
#         num_levels=num_levels,
#         temperature=temperature
#     )
    
#     rq_transformer = ImprovedRQTransformer(
#         d_model=512,
#         code_embedding_dim=embedding_dim,
#         num_embeddings=num_embeddings,
#         num_quantizers=num_quantizers
#     )
    
#     # Optimizers
#     vqvae_optimizer = torch.optim.Adam(vqvae_model.parameters(), lr=learning_rate)
#     transformer_optimizer = torch.optim.Adam(rq_transformer.parameters(), lr=learning_rate)
    
#     # Train improved VQ-VAE first
#     print("Training Improved Laplacian VQ-VAE...")
#     vqvae_model = train_laplacian_vqvae(vqvae_model, dataloader, vqvae_optimizer, 
#                                         epochs=vqvae_epochs, device=device)
    
#     # Save the model
#     torch.save(vqvae_model.state_dict(), 'improved_laplacian_vqvae.pth')
    
#     # Visualize some reconstructions
#     test_data = next(iter(dataloader))[:4]
#     reconstruction_fig = visualize_reconstruction(vqvae_model, test_data, device=device)
#     reconstruction_fig.savefig('reconstructions.png')
    
#     # Visualize Laplacian pyramid for a sample
#     sample_pyramid = build_laplacian_pyramid(test_data[:1].to(device), num_levels=vqvae_model.num_levels-1)
#     pyramid_fig = visualize_laplacian_pyramid(sample_pyramid)
#     pyramid_fig.savefig('laplacian_pyramid.png')
    
#     # Train improved RQ Transformer
#     print("Training Improved RQ Transformer...")
#     rq_transformer = train_improved_rq_transformer(
#         rq_transformer, vqvae_model, dataloader, transformer_optimizer, 
#         epochs=transformer_epochs, device=device
#     )
    
#     # Save the transformer model
#     torch.save(rq_transformer.state_dict(), 'improved_rq_transformer.pth')
    
#     # Generate samples
#     print("Generating samples...")
#     samples = generate_images(rq_transformer, vqvae_model, num_images=4, device=device)
    
#     # Display samples
#     samples = (samples + 1) / 2.0  # Denormalize
#     samples = samples.cpu().permute(0, 2, 3, 1).numpy()
    
#     fig, axes = plt.subplots(2, 2, figsize=(10, 10))
#     for i, ax in enumerate(axes.flat):
#         if i < len(samples):
#             ax.imshow(np.clip(samples[i], 0, 1))
#             ax.axis('off')
    
#     plt.savefig('generated_samples.png')
#     plt.close()
    
#     # Generate interpolations between two test images
#     if test_data.shape[0] >= 2:
#         print("Generating interpolations...")
#         interpolations = interpolate_in_latent_space(
#             vqvae_model, test_data[0], test_data[1], steps=8, device=device
#         )
        
#         # Display interpolations
#         interp_fig, axes = plt.subplots(1, 8, figsize=(16, 3))
#         for i, ax in enumerate(axes):
#             img = interpolations[i].permute(1, 2, 0).numpy()
#             ax.imshow(np.clip(img, 0, 1))
#             ax.axis('off')
        
#         plt.savefig('interpolations.png')
#         plt.close()
    
#     print("Done!")

if __name__ == "__main__":
    import torch
    import torchvision
    from torchvision import datasets, transforms
    import matplotlib.pyplot as plt
    
    # Hyperparameters
    batch_size = 32
    vqvae_epochs = 15
    transformer_epochs = 10
    learning_rate = 3e-4
    num_embeddings = 1024
    embedding_dim = 64
    num_quantizers = 4
    num_levels = 3
    temperature = 0.5  # Temperature for soft sampling
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Dataset
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Use any dataset
    dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    # Create improved models
    vqvae_model = ImprovedLaplacianVQVAE(
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        num_quantizers=num_quantizers,
        num_levels=num_levels,
        temperature=temperature
    )
    
    # Optimizers
    vqvae_optimizer = torch.optim.Adam(vqvae_model.parameters(), lr=learning_rate)
    
    # Train improved VQ-VAE first
    print("Training Improved Laplacian VQ-VAE...")
    vqvae_model = train_laplacian_vqvae(vqvae_model, dataloader, vqvae_optimizer, 
                                        epochs=vqvae_epochs, device=device)
    
    # Save the model
    torch.save(vqvae_model.state_dict(), 'improved_laplacian_vqvae.pth')
    
    # Visualize some reconstructions
    test_data = next(iter(dataloader))[:4]
    reconstruction_fig = visualize_reconstruction(vqvae_model, test_data, device=device)
    reconstruction_fig.savefig('reconstructions.png')
    
    # Visualize Laplacian pyramid for a sample
    sample_pyramid = build_laplacian_pyramid(test_data[:1].to(device), num_levels=vqvae_model.num_levels-1)
    pyramid_fig = visualize_laplacian_pyramid(sample_pyramid)
    pyramid_fig.savefig('laplacian_pyramid.png')
    
    # Create a simplified transformer that still uses spatial and depth components
    class StandardSpatialDepthTransformer(nn.Module):
        def __init__(self, d_model=512, spatial_nhead=8, depth_nhead=4,
                    spatial_layers=6, depth_layers=2,
                    code_embedding_dim=64, num_embeddings=1024, 
                    max_depth_positions=4, max_seq_len=64):
            super(StandardSpatialDepthTransformer, self).__init__()
            
            self.num_embeddings = num_embeddings
            self.max_depth_positions = max_depth_positions
            
            # Spatial transformer for modeling spatial dependencies
            self.spatial_transformer = SpatialTransformer(
                d_model=d_model,
                nhead=spatial_nhead,
                num_layers=spatial_layers,
                code_embedding_dim=code_embedding_dim,
                num_embeddings=num_embeddings,
                max_len=max_seq_len
            )
            
            # Depth transformer for predicting tokens at different depths
            self.depth_transformer = DepthTransformer(
                d_model=d_model,
                nhead=depth_nhead,
                num_layers=depth_layers,
                num_embeddings=num_embeddings,
                max_depth_positions=max_depth_positions
            )
            
            # Start-of-sequence token embedding
            self.sos_embedding = nn.Parameter(torch.randn(1, 1, code_embedding_dim))
            
        def forward(self, code_embeddings, target_indices=None, generation=False):
            batch_size = code_embeddings.shape[0]
            
            # Add SOS token at the beginning
            code_embeddings = torch.cat([
                self.sos_embedding.expand(batch_size, -1, -1),
                code_embeddings
            ], dim=1)
            
            # Create causal mask for spatial transformer
            seq_len = code_embeddings.shape[1]
            spatial_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=code_embeddings.device) * float('-inf'),
                diagonal=1
            )
            
            # Run through spatial transformer
            spatial_output = self.spatial_transformer(code_embeddings, spatial_mask)
            
            if generation:
                # For inference, return just the last position's output
                last_position = spatial_output[:, -1:, :]
                
                # Initialize with start tokens
                current_depth_tokens = torch.zeros(
                    (batch_size, 1), dtype=torch.long, device=code_embeddings.device
                )
                
                all_depth_tokens = []
                
                # Generate one token at a time for each depth
                for d in range(self.max_depth_positions):
                    # Create causal mask for depth
                    depth_len = current_depth_tokens.shape[1]
                    depth_mask = torch.triu(
                        torch.ones(depth_len, depth_len, device=code_embeddings.device) * float('-inf'),
                        diagonal=1
                    )
                    
                    # Predict next token
                    depth_logits = self.depth_transformer(
                        current_depth_tokens, last_position, depth_mask
                    )
                    
                    # Get most likely token
                    next_token = torch.argmax(depth_logits[:, -1, :], dim=-1, keepdim=True)
                    
                    # Add to sequence
                    current_depth_tokens = torch.cat([current_depth_tokens, next_token], dim=1)
                    all_depth_tokens.append(next_token)
                
                # Return the predicted depth tokens (excluding start token)
                return torch.cat(all_depth_tokens, dim=1)
            
            else:
                # For training
                outputs = []
                
                for i in range(1, seq_len):
                    current_position = spatial_output[:, i:i+1, :]
                    
                    # Get target indices for this position
                    target_for_position = target_indices[:, i-1, :]
                    
                    # Initialize with start token
                    current_depth_tokens = torch.zeros(
                        (batch_size, 1), dtype=torch.long, device=code_embeddings.device
                    )
                    
                    # For each depth level
                    for d in range(self.max_depth_positions):
                        # Add previous token for teacher forcing
                        if d > 0:
                            current_depth_tokens = torch.cat([
                                current_depth_tokens, 
                                target_for_position[:, d-1:d]
                            ], dim=1)
                        
                        # Create mask for depth
                        depth_len = current_depth_tokens.shape[1]
                        depth_mask = torch.triu(
                            torch.ones(depth_len, depth_len, device=code_embeddings.device) * float('-inf'),
                            diagonal=1
                        )
                        
                        # Predict next token
                        depth_logits = self.depth_transformer(
                            current_depth_tokens, current_position, depth_mask
                        )
                        
                        # Get prediction for this depth
                        pred = depth_logits[:, -1:, :]
                        outputs.append(pred)
                
                # Stack all predictions
                return torch.stack(outputs, dim=1)
    
    # Create the standard spatial-depth transformer
    standard_transformer = StandardSpatialDepthTransformer(
        d_model=512,
        spatial_nhead=8,
        depth_nhead=4,
        spatial_layers=6,
        depth_layers=2,
        code_embedding_dim=embedding_dim,
        num_embeddings=num_embeddings,
        max_depth_positions=num_quantizers  # Use the same value but with new parameter name
    )
    
    # Train transformer
    print("Training Standard Spatial-Depth Transformer...")
    transformer_optimizer = torch.optim.Adam(standard_transformer.parameters(), lr=learning_rate)
    standard_transformer = train_rq_transformer(  # We can reuse this training function
        standard_transformer, vqvae_model, dataloader, transformer_optimizer, 
        epochs=transformer_epochs, device=device
    )
    
    # Save the transformer model
    torch.save(standard_transformer.state_dict(), 'standard_spatial_depth_transformer.pth')
    
    # Generate samples
    print("Generating samples...")
    samples = generate_images(  # We can reuse this generation function too
        standard_transformer, vqvae_model, num_images=4, device=device
    )
    
    # Display samples
    samples = (samples + 1) / 2.0  # Denormalize
    samples = samples.cpu().permute(0, 2, 3, 1).numpy()
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        if i < len(samples):
            ax.imshow(np.clip(samples[i], 0, 1))
            ax.axis('off')
    
    plt.savefig('generated_samples.png')
    plt.close()
    
    print("Done!")
