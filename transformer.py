import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from torchvision.datasets import CIFAR10  # We'll use CIFAR10 for simplicity

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Import the VQVAE model directly
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

class WaveletVQVAE(nn.Module):
    """
    Simplified VQ-VAE (without explicit wavelet dependencies)
    """
    def __init__(self, num_embeddings=1024, embedding_dim=64, num_levels=4):
        super().__init__()
        self.num_levels = num_levels

        # Encoders for each level
        self.encoders = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
                nn.ReLU(True),
                nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
                nn.ReLU(True),
                nn.Conv2d(128, embedding_dim, kernel_size=3, stride=1, padding=1),
            ) for _ in range(num_levels)
        ])

        # Vector Quantizers for each level
        self.quantizers = nn.ModuleList([
            VectorQuantizer(num_embeddings, embedding_dim) for _ in range(num_levels)
        ])

        # Decoders for each level
        self.decoders = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(embedding_dim, 128, kernel_size=4, stride=2, padding=1, output_padding=1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, output_padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
            ) for _ in range(num_levels)
        ])
    
    def forward(self, x):
        """
        Simplified forward pass that works with a multi-resolution approach
        
        Args:
            x: Input image tensor [B, C, H, W]
            
        Returns:
            reconstructed: List of reconstructed images at different scales
            total_loss: Total VQ loss
            indices_list: List of quantized indices for each level
        """
        # Original resolution is defined as "level 0"
        inputs = [x]
        # Create downsampled versions for the other levels
        for i in range(1, self.num_levels):
            inputs.append(F.interpolate(x, scale_factor=1/(2**i), mode='bilinear', align_corners=False))

        reconstructed = []
        total_loss = 0
        indices_list = []

        for i in range(self.num_levels):
            # Encode
            encoded = self.encoders[i](inputs[i])
            
            # Quantize
            quantized, loss, indices = self.quantizers[i](encoded)
            total_loss += loss
            indices_list.append(indices)
            
            # Decode
            reconstructed_i = self.decoders[i](quantized)
            
            # Resize back to original resolution if needed
            if i > 0:
                reconstructed_i = F.interpolate(reconstructed_i, size=inputs[0].shape[2:], mode='bilinear', align_corners=False)
            
            reconstructed.append(reconstructed_i)

        return reconstructed, total_loss, indices_list

class PositionalEncoding(nn.Module):
    """
    Positional encoding for the transformer models
    """
    def __init__(self, d_model, max_len=10000):  # Increased max_len
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

class StandardTransformer(nn.Module):
    """
    Standard Transformer model for image generation using VQ-VAE tokens
    """
    def __init__(self, d_model=512, nhead=8, num_layers=6, 
                 dim_feedforward=2048, dropout=0.1, 
                 code_embedding_dim=64, num_embeddings=1024, max_len=10000):  # Increased max_len
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

def get_token_indices_from_vqvae(vqvae_model, images, device=device):
    """
    Extract token indices from the trained VQ-VAE
    
    Args:
        vqvae_model: Trained VQ-VAE model
        images: Batch of images [B, C, H, W]
        
    Returns:
        List of indices for each level
    """
    with torch.no_grad():
        # Process image at multiple scales
        scales = [images]  # Start with original scale
        for i in range(1, vqvae_model.num_levels):
            # Create downsampled versions
            scaled = F.interpolate(images, scale_factor=1/(2**i), mode='bilinear', align_corners=False)
            scales.append(scaled)
        
        indices_list = []
        for i, scaled_img in enumerate(scales):
            # Encode
            encoded = vqvae_model.encoders[i](scaled_img)
            
            # Quantize (only need the indices)
            _, _, indices = vqvae_model.quantizers[i](encoded)
            indices_list.append(indices)
            
    return indices_list

def train_standard_transformer(transformer_model, vqvae_model, train_data, 
                               optimizer, criterion=None, epochs=10, 
                               batch_size=8, device=device):
    """
    Train the standard transformer model with manual batching to avoid multiprocessing issues
    """
    transformer_model.to(device)
    vqvae_model.to(device)
    transformer_model.train()
    vqvae_model.eval()  # Freeze VQ-VAE
    
    # Default criterion
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    
    # Number of batches
    num_samples = len(train_data)
    num_batches = (num_samples + batch_size - 1) // batch_size  # Ceiling division
    epoch_losses = []
    for epoch in range(epochs):

        total_loss = 0
        # Shuffle indices for this epoch
        indices = torch.randperm(num_samples)
        
        loop = tqdm(range(num_batches), desc=f"Epoch {epoch}", leave=False)
        
        for batch_idx in loop:
            # Get batch indices
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_samples)
            batch_indices = indices[start_idx:end_idx]
            
            # Form batch
            images = torch.stack([train_data[i][0] for i in batch_indices]).to(device)
            
            # Get tokenized representation from VQ-VAE
            token_indices = get_token_indices_from_vqvae(vqvae_model, images, device)
            
            # Calculate sequence length for each token set
            seq_lengths = [indices.numel() // images.shape[0] for indices in token_indices]
            total_seq_len = sum(seq_lengths)
            
            # Print sequence length information for the first batch
            if batch_idx == 0:
                print(f"Sequence lengths: {seq_lengths}, Total: {total_seq_len}")
            
            # For very long sequences, sample a subset for training
            max_train_seq = 2048  # Maximum sequence length to use in training
            
            # Flatten and combine indices for all levels
            flattened_tokens = []
            for level_indices in token_indices:
                # Flatten spatial dimensions
                flat_tokens = level_indices.reshape(images.shape[0], -1)
                flattened_tokens.append(flat_tokens)
            
            # Concatenate all tokens into one sequence per batch
            token_sequences = torch.cat(flattened_tokens, dim=1)
            
            # If sequence is too long, subsample
            if token_sequences.shape[1] > max_train_seq:
                token_sequences = token_sequences[:, :max_train_seq]
                if batch_idx == 0:
                    print(f"Reduced sequence length to {token_sequences.shape[1]}")
            
            # Create input and target sequences for next-token prediction
            src_tokens = token_sequences[:, :-1]
            tgt_tokens = token_sequences[:, 1:]
            
            # Forward pass
            logits = transformer_model(src_tokens)
            
            # Compute loss
            logits_flat = logits.reshape(-1, transformer_model.num_embeddings)
            targets_flat = tgt_tokens.reshape(-1)
            
            loss = criterion(logits_flat, targets_flat)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Log training progress
            loop.set_postfix(loss=loss.item())
            
            if batch_idx % 10 == 0:
                print(f'Epoch: {epoch}, Batch: {batch_idx}/{num_batches}, Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / num_batches
        print(f'Epoch: {epoch}, Average Loss: {avg_loss:.4f}')
        epoch_losses.append(avg_loss)
        
        # Save checkpoint after each epoch
        os.makedirs('checkpoints', exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': transformer_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, f'checkpoints/wavelet_transformer_epoch_{epoch}.pt')
    
    # Save final model
    torch.save(transformer_model.state_dict(), 'checkpoints/wavelet_transformer_final.pt')
    return transformer_model, epoch_losses

def generate_images_with_transformer(transformer_model, vqvae_model, num_images=4, device=device):
    """
    Generate images using the trained transformer and VQ-VAE
    """
    transformer_model.eval()
    vqvae_model.eval()
    
    with torch.no_grad():
        batch_size = num_images
        
        # For demonstration, generate a small sequence
        gen_seq_len = 1024  
        
        # Start with a single start token (0)
        generated_sequence = torch.zeros((batch_size, 1), dtype=torch.long, device=device)
        
        # Generate tokens autoregressively
        for i in range(gen_seq_len):
            # Predict next token
            next_token = transformer_model(generated_sequence, generation=True)
            
            # Add to sequence
            generated_sequence = torch.cat([generated_sequence, next_token], dim=1)
            
            if i % 100 == 0:
                print(f"Generated {i+1}/{gen_seq_len} tokens")
        
        # Remove the initial start token
        generated_sequence = generated_sequence[:, 1:]
        
        # Reshape based on the assumed spatial dimensions
        size = int(np.sqrt(gen_seq_len))  # Assume square spatial layout
        
        # Adjust dimensions if needed
        level_tokens = generated_sequence[:, :size*size].reshape(batch_size, size, size)
        
        # Convert tokens to quantized representations
        flattened_tokens = level_tokens.reshape(batch_size, -1)
        
        # Create quantized representation using the code embeddings
        quantized = torch.zeros(batch_size, vqvae_model.quantizers[0].embedding_dim, size, size, device=device)
        
        # Get embeddings for these tokens
        embeddings = vqvae_model.quantizers[0].embeddings(flattened_tokens)
        embeddings = embeddings.reshape(batch_size, size, size, -1).permute(0, 3, 1, 2)
        quantized = embeddings
        
        # Decode this level
        generated_images = vqvae_model.decoders[0](quantized)
        
        # Denormalize
        generated_images = (generated_images + 1) / 2.0
        
        # Move to CPU for visualization
        generated_images = generated_images.cpu()
        
        # Visualize
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        for i, ax in enumerate(axes.flat):
            if i < num_images:
                ax.imshow(generated_images[i].permute(1, 2, 0).clamp(0, 1).numpy())
                ax.set_title(f'Generated {i+1}')
                ax.axis('off')
        
        plt.tight_layout()
        plt.savefig('generated_images.png')
        plt.show()
        
        return generated_images

def main():
    # Hyperparameters
    batch_size = 8
    transformer_epochs = 20
    learning_rate = 1e-4
    num_embeddings = 256
    embedding_dim = 64
    num_levels = 4
    
    # Load the trained VQ-VAE model
    vqvae_model = WaveletVQVAE(
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        num_levels=num_levels
    )
    
    try:
        vqvae_model.load_state_dict(torch.load('wavelet_vqvae.pth', map_location=device))
        print("Successfully loaded VQ-VAE model")
    except Exception as e:
        print(f"Error loading VQ-VAE model: {e}")
        print("Training will continue with a randomly initialized VQ-VAE, but results may not be optimal.")
    
    vqvae_model.to(device)
    vqvae_model.eval()
    
    # Create the transformer model
    transformer_model = StandardTransformer(
        d_model=512,
        nhead=8,
        num_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        code_embedding_dim=embedding_dim,
        num_embeddings=num_embeddings,
        max_len=10000
    )
    
    # Load CIFAR10 dataset (for simplicity and compatibility)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    
    train_data = CIFAR10(root="./data", train=True, download=True, transform=transform)
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(transformer_model.parameters(), lr=learning_rate)
    
    # Train the transformer with manual batching
    print("Training Standard Transformer...")
    transformer_model, epoch_losses = train_standard_transformer(
        transformer_model,
        vqvae_model,
        train_data,
        optimizer,
        epochs=transformer_epochs,
        batch_size=batch_size,
        device=device
    )
    
    # Generate sample images
    print("Generating sample images...")
    generate_images_with_transformer(transformer_model, vqvae_model, num_images=4)
    plt.figure(figsize=(8, 5))
    plt.plot(epoch_losses, marker='o')
    plt.title("Transformer Training Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig("transformer_loss_plot.png")
    plt.show()
    print("Done!")

if __name__ == "__main__":
    main()