import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os

from rqvae import ImprovedLaplacianVQVAE, LaplacianVQVAE, RQTransformer, build_laplacian_pyramid, reconstruct_from_laplacian_pyramid

# First check if the code file containing the model definitions is imported
# If not, we'll assume all model definitions are in the current scope
# from your_code_file import *  # Uncomment if model definitions are in a separate file

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Path to checkpoints
checkpoint_dir = "checkpoints"
checkpoint_files = {
    "vqvae_l2": os.path.join(checkpoint_dir, "wavelet_vqvae-l2.pth"),
    "vqvae_original": os.path.join(checkpoint_dir, "wavelet_vq...original.pth"),
    "vqvae_nossim": os.path.join(checkpoint_dir, "wavelet_vq...nossim.pth"),
    "vqvae_overall": os.path.join(checkpoint_dir, "wavelet_vq...-overall.pth"),
    "vqvae_ssim": os.path.join(checkpoint_dir, "wavelet_vq...o-ssim.pth")
}

def load_vqvae_model(checkpoint_path, model_type='improved'):
    """
    Load a VQ-VAE model from a checkpoint file
    
    Args:
        checkpoint_path: Path to the checkpoint file
        model_type: Type of model to load ('improved' or 'standard')
        
    Returns:
        Loaded model
    """
    # Create model based on type
    if model_type == 'improved':
        model = ImprovedLaplacianVQVAE(
            num_embeddings=1024,
            embedding_dim=64,
            num_quantizers=4,
            num_levels=3,
            temperature=0.5
        )
    else:
        model = LaplacianVQVAE(
            num_embeddings=1024,
            embedding_dim=64,
            num_quantizers=4,
            num_levels=3
        )
    
    # Load checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Check if it's a state_dict directly or contained within checkpoint
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
            
        model.load_state_dict(state_dict)
        print(f"Successfully loaded checkpoint from {checkpoint_path}")
    except Exception as e:
        print(f"Error loading checkpoint from {checkpoint_path}: {e}")
        
        # Print model's state_dict keys for debugging
        print("Model state_dict keys:")
        for k in model.state_dict().keys():
            print(f"  {k}")
            
        # Print checkpoint keys for debugging
        if isinstance(checkpoint, dict):
            print("Checkpoint keys:")
            for k in checkpoint.keys():
                print(f"  {k}")
    
    model.to(device)
    model.eval()
    return model

def load_transformer_model(checkpoint_path, vqvae_model):
    """
    Load a transformer model for the VQ-VAE
    
    Args:
        checkpoint_path: Path to the checkpoint file
        vqvae_model: The VQ-VAE model for which this transformer was trained
    
    Returns:
        Loaded transformer model
    """
    # Determine if we should use the improved transformer
    if isinstance(vqvae_model, ImprovedLaplacianVQVAE):
        num_quantizers = len(vqvae_model.quantizers)
        embedding_dim = vqvae_model.quantizers[0].embedding_dim
        num_embeddings = vqvae_model.quantizers[0].num_embeddings
        
        model = RQTransformer(
            d_model=512,
            spatial_nhead=8,
            depth_nhead=4,
            spatial_layers=6,
            depth_layers=2,
            code_embedding_dim=embedding_dim,
            num_embeddings=num_embeddings,
            num_quantizers=4  # This should match what was used in training
        )
    else:
        # For standard LaplacianVQVAE
        num_quantizers = 4  # Assuming 4 quantizers as in your code
        embedding_dim = vqvae_model.quantizers[0].quantizers[0].embedding_dim
        num_embeddings = vqvae_model.quantizers[0].quantizers[0].num_embeddings
        
        model = RQTransformer(
            d_model=512,
            spatial_nhead=8,
            depth_nhead=4,
            spatial_layers=6,
            depth_layers=2,
            code_embedding_dim=embedding_dim,
            num_embeddings=num_embeddings,
            num_quantizers=num_quantizers
        )
    
    # Load checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Check if it's a state_dict directly or contained within checkpoint
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
            
        model.load_state_dict(state_dict)
        print(f"Successfully loaded transformer from {checkpoint_path}")
    except Exception as e:
        print(f"Error loading transformer from {checkpoint_path}: {e}")
        # We could add the same debug printing as above
    
    model.to(device)
    model.eval()
    return model

def visualize_reconstruction(vqvae_model, input_images):
    """
    Reconstruct and visualize images using the VQ-VAE
    
    Args:
        vqvae_model: Trained VQ-VAE model
        input_images: Images to reconstruct
    """
    vqvae_model.eval()
    with torch.no_grad():
        # Move inputs to the correct device
        input_images = input_images.to(device)
        
        # Build Laplacian pyramid
        laplacian_levels = build_laplacian_pyramid(input_images, num_levels=vqvae_model.num_levels-1)
        
        # Forward pass
        if isinstance(vqvae_model, ImprovedLaplacianVQVAE):
            reconstructed_levels, _, _, _ = vqvae_model(laplacian_levels, training=False)
        else:
            reconstructed_levels, _, _ = vqvae_model(laplacian_levels)
        
        # Reconstruct from Laplacian pyramid
        reconstructed = reconstruct_from_laplacian_pyramid(reconstructed_levels)
        
        # Denormalize for visualization
        input_images = (input_images + 1) / 2.0
        reconstructed = (reconstructed + 1) / 2.0
        
        # Move to CPU for visualization
        input_images = input_images.cpu()
        reconstructed = reconstructed.cpu()
        
        # Visualize
        num_images = min(4, input_images.shape[0])
        fig, axes = plt.subplots(2, num_images, figsize=(num_images * 3, 6))
        
        for i in range(num_images):
            # Original
            if num_images == 1:
                axes[0].imshow(input_images[i].permute(1, 2, 0).numpy())
                axes[0].set_title('Original')
                axes[0].axis('off')
                
                # Reconstructed
                axes[1].imshow(reconstructed[i].permute(1, 2, 0).numpy())
                axes[1].set_title('Reconstructed')
                axes[1].axis('off')
            else:
                axes[0, i].imshow(input_images[i].permute(1, 2, 0).numpy())
                axes[0, i].set_title('Original')
                axes[0, i].axis('off')
                
                # Reconstructed
                axes[1, i].imshow(reconstructed[i].permute(1, 2, 0).numpy())
                axes[1, i].set_title('Reconstructed')
                axes[1, i].axis('off')
        
        plt.tight_layout()
        plt.savefig('reconstructions.png')
        plt.show()
        
        return reconstructed

def generate_images(transformer_model, vqvae_model, num_images=4, seq_len=64):
    """
    Generate images using the transformer and VQ-VAE
    
    Args:
        transformer_model: Trained transformer model
        vqvae_model: Trained VQ-VAE model
        num_images: Number of images to generate
        seq_len: Length of sequence to generate
    
    Returns:
        Generated images
    """
    transformer_model.eval()
    vqvae_model.eval()
    
    with torch.no_grad():
        # Initialize with empty embeddings
        batch_size = num_images
        
        # Get embedding dimension from the vqvae model
        if isinstance(vqvae_model, ImprovedLaplacianVQVAE):
            code_embedding_dim = vqvae_model.quantizers[0].embedding_dim
        else:
            code_embedding_dim = vqvae_model.quantizers[0].quantizers[0].embedding_dim
        
        # Start with empty embeddings
        code_embeddings = torch.zeros(batch_size, 0, code_embedding_dim, device=device)
        
        # Generate tokens autoregressively
        all_tokens = []
        
        for i in range(seq_len):
            # Predict next position
            next_tokens = transformer_model(code_embeddings, generation=True)
            all_tokens.append(next_tokens)
            
            # Get embeddings for the predicted tokens
            next_embedding = torch.zeros(batch_size, code_embedding_dim, device=device)
            
            # Different embedding access based on model type
            if isinstance(vqvae_model, ImprovedLaplacianVQVAE):
                for d, tokens in enumerate(next_tokens.unbind(-1)):
                    next_embedding += vqvae_model.quantizers[0].codebook(tokens)
            else:
                for d, tokens in enumerate(next_tokens.unbind(-1)):
                    next_embedding += vqvae_model.quantizers[0].quantizers[d].embeddings(tokens)
            
            # Add to code embeddings
            code_embeddings = torch.cat([
                code_embeddings, 
                next_embedding.unsqueeze(1)
            ], dim=1)
        
        # Stack all tokens
        all_tokens = torch.stack(all_tokens, dim=1)  # [batch, seq_len, depth]
        
        # Reshape to spatial dimensions
        h = w = int(np.sqrt(seq_len))  # Assuming square spatial dimensions
        
        # Organize tokens by level and depth
        tokens_per_level = seq_len // vqvae_model.num_levels
        level_h = level_w = int(np.sqrt(tokens_per_level))
        
        # Process each level
        reconstructed_levels = []
        
        for level in range(vqvae_model.num_levels):
            start_idx = level * tokens_per_level
            end_idx = (level + 1) * tokens_per_level
            
            level_tokens = all_tokens[:, start_idx:end_idx, :]
            level_tokens = level_tokens.reshape(batch_size, level_h, level_w, -1)
            
            # Convert tokens to quantized representations
            if isinstance(vqvae_model, ImprovedLaplacianVQVAE):
                quantized = torch.zeros(batch_size, code_embedding_dim, level_h, level_w, device=device)
                for d in range(level_tokens.shape[-1]):
                    tokens = level_tokens[:, :, :, d].reshape(batch_size, -1)
                    embeddings = vqvae_model.quantizers[level].codebook(tokens)
                    embeddings = embeddings.reshape(batch_size, level_h, level_w, -1).permute(0, 3, 1, 2)
                    quantized += embeddings
            else:
                quantized = torch.zeros(batch_size, code_embedding_dim, level_h, level_w, device=device)
                for d in range(level_tokens.shape[-1]):
                    tokens = level_tokens[:, :, :, d].reshape(batch_size, -1)
                    embeddings = vqvae_model.quantizers[level].quantizers[d].embeddings(tokens)
                    embeddings = embeddings.reshape(batch_size, level_h, level_w, -1).permute(0, 3, 1, 2)
                    quantized += embeddings
            
            # Decode this level
            reconstructed = vqvae_model.decoders[level](quantized)
            reconstructed_levels.append(reconstructed)
        
        # Reconstruct from Laplacian pyramid
        generated_images = reconstruct_from_laplacian_pyramid(reconstructed_levels)
        
        # Denormalize
        generated_images = (generated_images + 1) / 2.0
        
        # Move to CPU
        generated_images = generated_images.cpu()
        
        # Visualize
        fig, axes = plt.subplots(2, 2, figsize=(8, 8))
        for i, ax in enumerate(axes.flat):
            if i < num_images:
                ax.imshow(generated_images[i].permute(1, 2, 0).numpy())
                ax.set_title(f'Generated {i+1}')
                ax.axis('off')
        
        plt.tight_layout()
        plt.savefig('generated_images.png')
        plt.show()
        
        return generated_images

# Main execution
if __name__ == "__main__":
    # Example usage:
    # 1. Load a VQ-VAE model
    vqvae_checkpoint = checkpoint_files["vqvae_l2"]  # Choose the checkpoint you want to use
    vqvae_model = load_vqvae_model(vqvae_checkpoint, model_type='improved')
    
    # 2. Load a transformer model (if available)
    transformer_checkpoint = checkpoint_files["vqvae_original"]  # Replace with the actual transformer checkpoint
    transformer_model = load_transformer_model(transformer_checkpoint, vqvae_model)
    
    # 3. Test with some images
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Use a sample dataset (replace with your own dataset if you have one)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=True)
    
    # Get a batch of images
    test_images, _ = next(iter(test_loader))
    
    # 4. Visualize reconstructions
    print("Visualizing reconstructions...")
    reconstructed = visualize_reconstruction(vqvae_model, test_images)
    
    # 5. Generate new images with the transformer (if available)
    print("Generating new images...")
    generated = generate_images(transformer_model, vqvae_model, num_images=4)
    
    print("Done!")