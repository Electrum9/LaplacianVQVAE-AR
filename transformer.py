import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm
import pywt
import ptwt
from vqvae import WaveletVQVAE, get_wavelet_coeffs
from torch.utils.tensorboard import SummaryWriter

class WaveletTransformerDataset(Dataset):
    """
    Dataset for training the autoregressive transformer on wavelet VQ-VAE indices
    """
    def __init__(self, dataloader, vqvae_model, device='cuda'):
        self.indices_data = []
        self.vqvae = vqvae_model.to(device)
        self.vqvae.eval()
        self.device = device
        
        print("Extracting indices from VQ-VAE...")
        with torch.no_grad():
            for batch in tqdm(dataloader):
                # Get images from dataloader
                images = batch['image'].to(device)
                
                # Get wavelet coefficients
                wavelet_coeffs = get_wavelet_coeffs(images)
                
                # Extract quantized indices from VQ-VAE
                _, _, all_indices = self.vqvae(wavelet_coeffs)
                
                # Store indices for each batch
                for batch_idx in range(images.shape[0]):
                    indices_per_level = []
                    for level_idx in range(len(all_indices)):
                        indices_per_level.append(all_indices[level_idx][batch_idx].cpu())
                    self.indices_data.append(indices_per_level)
        
        print(f"Created dataset with {len(self.indices_data)} samples")
    
    def __len__(self):
        return len(self.indices_data)
    
    def __getitem__(self, idx):
        return self.indices_data[idx]


class WaveletARTransformer(nn.Module):
    """
    Autoregressive Transformer for generating wavelet coefficient tokens
    """
    def __init__(self, 
                 num_embeddings=256, 
                 embedding_dim=512, 
                 num_heads=8, 
                 num_layers=6, 
                 dropout=0.1,
                 num_levels=4):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.num_levels = num_levels
        
        # Token embedding for each codebook index
        self.token_embedding = nn.Embedding(num_embeddings, embedding_dim)
        
        # Special tokens for each wavelet coefficient type (LL, LH, HL, HH)
        self.coefficient_type_embedding = nn.Embedding(num_levels, embedding_dim)
        
        # Position embedding for spatial positions
        self.max_sequence_length = 4096  # Adjust based on expected image size and levels
        self.position_embedding = nn.Embedding(self.max_sequence_length, embedding_dim)
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim*4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # Output projection to predict next token
        self.output_projection = nn.Linear(embedding_dim, num_embeddings)
        
        # Sequence ordering based on frequency bands
        # Ordering tokens from lowest frequency (LL) to highest (HH)
        self.register_buffer('sequence_order', torch.tensor([0, 1, 2, 3]))
    
    def get_position_ids(self, token_indices):
        """
        Create position indices for the flattened wavelet coefficients
        """
        batch_size = token_indices[0].shape[0]
        position_ids = []
        
        position_offset = 0
        for level_idx in range(self.num_levels):
            level_shape = token_indices[level_idx].shape
            num_positions = level_shape[1] * level_shape[2]
            
            level_position_ids = torch.arange(
                position_offset, 
                position_offset + num_positions, 
                device=token_indices[0].device
            ).expand(batch_size, -1)
            
            position_ids.append(level_position_ids.reshape(batch_size, -1))
            position_offset += num_positions
            
        return torch.cat(position_ids, dim=1)
    
    def forward(self, token_indices, target_indices=None):
        """
        Forward pass through the transformer
        
        Args:
            token_indices: List of token indices for each wavelet level [batch_size, height, width]
            target_indices: List of target indices for next token prediction
            
        Returns:
            logits: Predicted token logits
            loss: Cross-entropy loss if target_indices provided
        """
        batch_size = token_indices[0].shape[0]
        
        # Flatten and concatenate all token indices
        flattened_indices = []
        coefficient_type_ids = []
        
        for level_idx in range(self.num_levels):
            level_indices = token_indices[level_idx]
            flattened_level = level_indices.reshape(batch_size, -1)
            flattened_indices.append(flattened_level)
            
            # Add coefficient type IDs for this level
            level_length = flattened_level.shape[1]
            coefficient_type_ids.append(torch.full((batch_size, level_length), level_idx, device=level_indices.device))
        
        # Concatenate all tokens in order from low to high frequency
        concatenated_indices = torch.cat(flattened_indices, dim=1)
        concatenated_type_ids = torch.cat(coefficient_type_ids, dim=1)
        
        # Get position IDs
        position_ids = self.get_position_ids(token_indices)
        
        # Get embeddings
        token_embeddings = self.token_embedding(concatenated_indices)
        coeff_type_embeddings = self.coefficient_type_embedding(concatenated_type_ids)
        position_embeddings = self.position_embedding(position_ids)
        
        # Combine embeddings
        embeddings = token_embeddings + coeff_type_embeddings + position_embeddings
        
        # Create attention mask for causal attention
        seq_length = embeddings.shape[1]
        attn_mask = torch.triu(
            torch.ones(seq_length, seq_length, device=embeddings.device) * float('-inf'),
            diagonal=1
        )
        
        # Process through transformer
        transformer_output = self.transformer(embeddings, mask=attn_mask)
        
        # Predict next token
        logits = self.output_projection(transformer_output)
        
        # Calculate loss if target_indices provided
        loss = None
        if target_indices is not None:
            # Flatten and concatenate target indices
            flattened_targets = []
            for level_idx in range(self.num_levels):
                flattened_targets.append(target_indices[level_idx].reshape(batch_size, -1))
            
            # Concatenate and shift right for next token prediction
            concatenated_targets = torch.cat(flattened_targets, dim=1)
            
            # Compute cross-entropy loss
            loss = F.cross_entropy(
                logits.reshape(-1, self.num_embeddings), 
                concatenated_targets.reshape(-1)
            )
        
        return logits, loss
    
    def generate(self, vqvae_model, batch_size=1, device='cuda', temperature=1.0):
        """
        Generate new wavelet coefficients autoregressively
        
        Args:
            vqvae_model: Trained WaveletVQVAE model
            batch_size: Number of images to generate
            temperature: Temperature for sampling
            
        Returns:
            reconstructed_image: Generated image
        """
        self.eval()
        vqvae_model.eval()
        
        with torch.no_grad():
            # Initialize empty token sequences for each level
            token_sequences = []
            sequence_lengths = []
            
            # Determine sequence lengths for each level based on typical image size
            # For simplicity, assume square images and power-of-2 dimensions
            base_size = 8  # Base size for the smallest feature maps
            for level_idx in range(self.num_levels):
                # Size increases as we go to lower frequencies
                # Adjust these calculations based on your specific architecture
                size = base_size * (2 ** (self.num_levels - level_idx - 1))
                sequence_lengths.append(size * size)
                
                # Initialize with zeros
                token_sequences.append(
                    torch.zeros((batch_size, size, size), dtype=torch.long, device=device)
                )
            
            # Total sequence length
            total_length = sum(sequence_lengths)
            
            # Generate tokens autoregressively
            for pos in tqdm(range(total_length), desc="Generating tokens"):
                # Determine which level and position we're currently generating
                level_idx = 0
                remaining_pos = pos
                
                while level_idx < self.num_levels - 1 and remaining_pos >= sequence_lengths[level_idx]:
                    remaining_pos -= sequence_lengths[level_idx]
                    level_idx += 1
                
                # Get spatial coordinates from flattened position
                size = int(np.sqrt(sequence_lengths[level_idx]))
                h, w = remaining_pos // size, remaining_pos % size
                
                # Forward pass to get logits
                logits, _ = self.forward(token_sequences)
                
                # Extract relevant logits
                flattened_indices = []
                for l_idx in range(self.num_levels):
                    flattened_indices.append(token_sequences[l_idx].reshape(batch_size, -1))
                
                concatenated_indices = torch.cat(flattened_indices, dim=1)
                relevant_pos = sum(sequence_lengths[:level_idx]) + (h * size + w)
                next_token_logits = logits[:, relevant_pos, :]
                
                # Sample next token
                if temperature == 0:
                    # Greedy sampling
                    next_token = torch.argmax(next_token_logits, dim=-1)
                else:
                    # Temperature sampling
                    probs = F.softmax(next_token_logits / temperature, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
                
                # Update token sequence
                token_sequences[level_idx][:, h, w] = next_token
            
            # Decode generated tokens using the VQ-VAE
            reconstructed_coeffs = []
            for level_idx in range(self.num_levels):
                # Use the quantizer directly to get embeddings
                indices = token_sequences[level_idx]
                batch_size, h, w = indices.shape
                
                # Get embeddings from indices
                flat_indices = indices.reshape(-1)
                embeddings = vqvae_model.quantizers[level_idx].embeddings(flat_indices)
                quantized = embeddings.reshape(batch_size, h, w, -1).permute(0, 3, 1, 2)
                
                # Decode using the corresponding decoder
                reconstructed = vqvae_model.decoders[level_idx](quantized)
                reconstructed_coeffs.append(reconstructed)
            
            # Reconstruct image from wavelet coefficients
            LL, LH, HL, HH = reconstructed_coeffs
            rc = (LL, (LH, HL, HH))
            reconstructed_image = ptwt.waverec2(rc, pywt.Wavelet("db8"))
            
            return reconstructed_image


def train_wavelet_transformer(model, dataset, optimizer, epochs=10, batch_size=32, device='cuda'):
    """
    Train the wavelet autoregressive transformer
    """
    writer = SummaryWriter(log_dir="runs/wavelet_transformer")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model.to(device)
    model.train()
    
    print("Training Wavelet AR Transformer...")
    for epoch in range(epochs):
        total_loss = 0
        loop = tqdm(dataloader, desc=f"Epoch {epoch}", leave=False)
        
        for batch_idx, data in enumerate(loop):
            # Each data item is a list of indices for each wavelet level
            token_indices = [indices.to(device) for indices in data]
            
            # Create target indices by offsetting (for autoregressive prediction)
            target_indices = []
            for level_indices in token_indices:
                # Shift left by 1 for next token prediction
                target = torch.roll(level_indices, shifts=-1, dims=2)
                # Mark the last column as padding (could use special token)
                target[:, :, -1] = 0
                target_indices.append(target)
            
            # Forward pass
            _, loss = model(token_indices, target_indices)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            loop.set_postfix(loss=loss.item())
            writer.add_scalar("Loss/train", loss.item(), batch_idx + epoch * len(dataloader))
            
            if batch_idx % 100 == 0:
                print(f"Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}")
        
        print(f"Epoch: {epoch}, Average Loss: {total_loss/len(dataloader):.4f}")
        
        # Sample and save generated images periodically
        if epoch % 5 == 0 or epoch == epochs - 1:
            model.eval()
            vqvae_model = WaveletVQVAE(num_embeddings=model.num_embeddings, embedding_dim=64, num_levels=model.num_levels)
            vqvae_model.load_state_dict(torch.load("wavelet_vqvae.pth"))
            vqvae_model.to(device)
            
            with torch.no_grad():
                generated_images = model.generate(vqvae_model, batch_size=4, device=device)
                for i in range(min(4, generated_images.shape[0])):
                    writer.add_image(f"Generated_Image_{i}", generated_images[i], epoch)
            
            model.train()
    
    return model


if __name__ == "__main__":
    import torchvision.transforms as transforms
    from datasets import load_dataset
    
    # Hyperparameters
    batch_size = 32
    transformer_epochs = 10
    learning_rate = 3e-4
    num_embeddings = 256
    embedding_dim = 512  # For transformer
    num_heads = 8
    num_layers = 6
    num_levels = 4
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    
    # Load the dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])
    
    dset = load_dataset("merkol/ffhq-256", split="train")
    dset.set_transform(lambda examples: {
        "image": torch.stack([transform(img) for img in examples["image"]])
    })
    dataloader = DataLoader(dset, batch_size=batch_size, shuffle=True)
    
    # Load trained VQ-VAE model
    vqvae_model = WaveletVQVAE(num_embeddings=num_embeddings, embedding_dim=64, num_levels=num_levels)
    vqvae_model.load_state_dict(torch.load("wavelet_vqvae.pth"))
    vqvae_model.eval()  # Set to evaluation mode
    
    # Create transformer dataset
    transformer_dataset = WaveletTransformerDataset(dataloader, vqvae_model, device=device)
    
    # Create transformer model
    transformer_model = WaveletARTransformer(
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        num_levels=num_levels
    )
    
    # Optimizer
    transformer_optimizer = torch.optim.Adam(transformer_model.parameters(), lr=learning_rate)
    
    # Train transformer model
    print("Training Wavelet Autoregressive Transformer...")
    transformer_model = train_wavelet_transformer(
        transformer_model,
        transformer_dataset,
        transformer_optimizer,
        epochs=transformer_epochs,
        batch_size=batch_size,
        device=device
    )
    
    # Save model
    torch.save(transformer_model.state_dict(), "wavelet_transformer.pth")
    
    # Generate samples
    print("Generating samples...")
    transformer_model.eval()
    with torch.no_grad():
        generated_images = transformer_model.generate(vqvae_model, batch_size=16, device=device)
    
    # Calculate reconstruction loss
    original_images = next(iter(dataloader))["image"].to(device)[:16]
    mse_loss = F.mse_loss(generated_images, original_images)
    l1_loss = F.l1_loss(generated_images, original_images)
    
    import torchmetrics
    ssim = torchmetrics.StructuralSimilarityIndexMeasure().to(device)
    ssim_score = ssim(generated_images, original_images)
    
    print(f"Generation Results:")
    print(f"MSE Loss: {mse_loss.item():.4f}")
    print(f"L1 Loss: {l1_loss.item():.4f}")
    print(f"SSIM Score: {ssim_score.item():.4f}")