from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import math
import ptwt
import pywt
from datasets import load_dataset
import torchmetrics

# Assuming vqvae.py contains the WaveletVQVAE class and get_wavelet_coeffs function
# Make sure vqvae.py is in the same directory or adjust the import path
try:
    from vqvae import WaveletVQVAE, get_wavelet_coeffs
except ImportError:
    print("Error: Could not import WaveletVQVAE or get_wavelet_coeffs from vqvae.py.")
    print("Ensure vqvae.py exists and is in the correct path.")
    exit()


# TensorBoard for logging
from torch.utils.tensorboard import SummaryWriter

# Metrics
ssim = torchmetrics.StructuralSimilarityIndexMeasure().to('cuda' if torch.cuda.is_available() else 'cpu')
wavelet = pywt.Wavelet("db8") # Ensure this matches the wavelet used in vqvae.py

# --- Helper Function for Data Transformation (Replaces Lambda) ---
# Define the transform globally so it can be pickled
global_transform = None # Will be initialized in main block

def apply_transform(examples):
    """
    Applies the global transform to a batch of images from the dataset.
    Needed for multiprocessing in DataLoader.
    """
    if global_transform is None:
        raise ValueError("Global transform not initialized. Run the main script block.")
    # Ensure images are RGB before transforming
    processed_images = [global_transform(img.convert("RGB")) for img in examples["image"]]
    return {"image": torch.stack(processed_images)}
# --------------------------------------------------------------------


class PositionalEncoding(nn.Module):
    """
    Standard positional encoding for Transformers.
    Adds sinusoidal positional information to input embeddings.
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1) # Shape: (max_len, 1, d_model)
        self.register_buffer('pe', pe) # Register as buffer so it's not a model parameter

    def forward(self, x):
        """
        Args:
            x: Input tensor (Sequence Length, Batch Size, Embedding Dim)
        Returns:
            Tensor with added positional encoding.
        """
        # Add positional encoding to the input embeddings
        # Ensure positional encoding length matches input sequence length
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class WaveletTransformer(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        num_heads: int,
        num_layers: int,
        hidden_dim: int,
        max_seq_len: int,
        quantized_shapes: List[Tuple[int,int,int]],
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len

        # quantized_shapes: [(B, H₀, W₀), (B, H₁, W₁), …]
        self.quantized_shapes = quantized_shapes
        # true full sequence length = Σ_i (H_i * W_i)
        self.full_len = sum(h * w for (_, h, w) in quantized_shapes)

        # token embedding + positional encoding
        self.token_embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.pos_encoder = PositionalEncoding(embedding_dim, dropout, max_seq_len)

        # split depth/spatial layers for the full path
        depth_layers = num_layers // 2
        spatial_layers = num_layers - depth_layers
        depth_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True,
            activation=F.gelu,
        )
        spatial_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True,
            activation=F.gelu,
        )

        self.depth_encoder = nn.TransformerEncoder(depth_layer, num_layers=depth_layers)
        self.spatial_encoder = nn.TransformerEncoder(spatial_layer, num_layers=spatial_layers)

        # tiny prefix encoder for short sequences
        prefix_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True,
            activation=F.gelu,
        )
        self.prefix_encoder = nn.TransformerEncoder(prefix_layer, num_layers=1)

        # final projection
        self.output_layer = nn.Linear(embedding_dim, num_embeddings)
        self._init_weights()

    def _init_weights(self):
        initrange = 0.1
        self.token_embedding.weight.data.uniform_(-initrange, initrange)
        self.output_layer.bias.data.zero_()
        self.output_layer.weight.data.uniform_(-initrange, initrange)


    def forward(self, indices: torch.Tensor):
        """
        indices: (B, S) where S may be short (<full_len, generation) or exactly full_len.
        """
        B, S = indices.shape

        # 1) embed + scale
        x = self.token_embedding(indices) * math.sqrt(self.embedding_dim)  # (B, S, D)
        # 2) add pos-encoding
        x = x.permute(1, 0, 2)                                             # (S, B, D)
        x = self.pos_encoder(x)
        x = x.permute(1, 0, 2)                                             # (B, S, D)

        # SHORT path: use the 1-layer prefix encoder for autoregressive gen
        if S < self.full_len:
            mask = nn.Transformer.generate_square_subsequent_mask(S).to(x.device)
            x = self.prefix_encoder(x, mask=mask)                          # (B, S, D)
            return self.output_layer(x)                                    # logits (B, S, V)

        # FULL “axial” path: reshape into (B, D, H*W, emb) and do depth→spatial
        # flatten all levels into one long sequence of length = full_len
        D = len(self.quantized_shapes)
        # assume for simplicity all levels share the same H,W but if they differ you can 
        # un-flatten individually using quantized_shapes
        _, H0, W0 = self.quantized_shapes[0]

        x = x.view(B, D, H0*W0, self.embedding_dim)                        # (B, D, H*W, D)
        # — depth mixing
        x = x.permute(0, 2, 1, 3).reshape(B * (H0*W0), D, -1)              # (B·H·W, D, D)
        x = self.depth_encoder(x)                                          # (B·H·W, D, D)
        x = x.view(B, H0*W0, D, self.embedding_dim).permute(0, 2, 1, 3)    # (B, D, H·W, D)

        # — spatial mixing (with causal mask per spatial block)
        x = x.reshape(B*D, H0*W0, self.embedding_dim)                     # (B·D, H·W, D)
        spatial_mask = nn.Transformer.generate_square_subsequent_mask(H0*W0).to(x.device)
        x = self.spatial_encoder(x, mask=spatial_mask)                    # (B·D, H·W, D)

        # back to flat and project
        x = x.view(B, D, H0*W0, self.embedding_dim).permute(0, 2, 1, 3)    # (B, H·W, D, D)
        x = x.reshape(B, S, self.embedding_dim)                            # (B, S, D)
        return self.output_layer(x)                                        # (B, S, V)



    @torch.no_grad()
    def generate(self, start_indices, max_len, temperature=1.0, top_k=None):
        """
        Autoregressively generate a sequence of indices.
        Args:
            start_indices (Tensor): Initial sequence of indices (Batch Size, Start Seq Len).
            max_len (int): Maximum length of the sequence to generate.
            temperature (float): Softmax temperature for sampling. Lower values make it more deterministic.
            top_k (int, optional): If set, sample only from the top k most likely next tokens.
        Returns:
            generated_indices (Tensor): Generated sequence of indices (Batch Size, max_len).
        """
        self.eval() # Set model to evaluation mode
        generated = start_indices
        batch_size = start_indices.size(0)

        for _ in tqdm(range(max_len - start_indices.size(1)), desc="Generating"):
            # Get the current sequence (ensure it doesn't exceed max_seq_len for positional encoding)
            current_indices = generated[:, -self.max_seq_len:] # Use only the last max_seq_len tokens

            # Get logits from the model
            logits = self(current_indices) # (Batch, Current Seq Len, Num Embeddings)

            # Focus only on the logits for the next step (last position)
            next_token_logits = logits[:, -1, :] # (Batch, Num Embeddings)

            # Apply temperature scaling
            next_token_logits = next_token_logits / temperature

            # Optional Top-K sampling
            if top_k is not None:
                v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                next_token_logits[next_token_logits < v[:, [-1]]] = -float('Inf')

            # Apply softmax to get probabilities
            probs = F.softmax(next_token_logits, dim=-1) # (Batch, Num Embeddings)

            # Sample the next token index
            next_token = torch.multinomial(probs, num_samples=1) # (Batch, 1)

            # Append the sampled token to the sequence
            generated = torch.cat((generated, next_token), dim=1)

        self.train() # Set model back to training mode
        return generated


# --- Training Function ---
def train_transformer(transformer_model, vqvae_model, dataloader, optimizer, criterion, epochs, device, writer, max_seq_len):
    """
    Train the Wavelet Transformer model.
    """
    transformer_model.to(device)
    vqvae_model.to(device)
    vqvae_model.eval() # Freeze VQ-VAE weights

    print("Training Wavelet Transformer...")
    global_step = 0
    quantized_shapes_epoch = None # Store shapes from the first batch of the first epoch

    for epoch in range(epochs):
        transformer_model.train() # Set transformer to training mode
        total_loss = 0
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=True)

        for batch_idx, data in enumerate(loop):
            # Check if data is a dictionary and has the 'image' key
            if isinstance(data, dict) and 'image' in data:
                images = data["image"].to(device)
            elif isinstance(data, torch.Tensor): # Handle cases where DataLoader might return only tensors
                 images = data.to(device)
            else:
                print(f"Warning: Unexpected data format received from DataLoader: {type(data)}. Skipping batch.")
                continue

            # Ensure images have 3 channels (handle potential grayscale)
            if images.shape[1] == 1:
                images = images.repeat(1, 3, 1, 1)

            with torch.no_grad(): # Don't need gradients for VQ-VAE part
                # 1. Get Wavelet Coefficients
                try:
                    wavelet_coeffs = get_wavelet_coeffs(images) # List [LL, LH, HL, HH]
                except Exception as e:
                    print(f"Error getting wavelet coeffs: {e}. Skipping batch.")
                    print(f"Image shape: {images.shape}")
                    continue


                # 2. Encode and Quantize using VQ-VAE to get indices
                all_indices = []
                quantized_shapes_batch = [] # Store shapes for this batch
                try:
                    for i in range(len(wavelet_coeffs)):
                        # Check if wavelet coeff tensor is valid
                        if not isinstance(wavelet_coeffs[i], torch.Tensor) or wavelet_coeffs[i].numel() == 0:
                             print(f"Warning: Invalid wavelet coefficient at index {i}. Skipping subband.")
                             continue

                        encoded = vqvae_model.encoders[i](wavelet_coeffs[i])
                        _, _, indices = vqvae_model.quantizers[i](encoded) # We only need indices
                        quantized_shapes_batch.append(indices.shape) # Store shape (B, H_i, W_i)
                        # Flatten indices: (B, H_i, W_i) -> (B, H_i * W_i)
                        all_indices.append(indices.view(images.size(0), -1))

                    # Store shapes from the first batch of the first epoch
                    if epoch == 0 and batch_idx == 0:
                        quantized_shapes_epoch = quantized_shapes_batch

                except Exception as e:
                    print(f"Error during VQ-VAE encoding/quantization: {e}. Skipping batch.")
                    continue

                # Check if any indices were generated
                if not all_indices:
                    print("Warning: No indices generated for this batch. Skipping.")
                    continue

                # 3. Concatenate indices from all subbands into a single sequence
                # Shape: (B, total_indices_length) where total_indices_length = sum(H_i * W_i)
                indices_sequence = torch.cat(all_indices, dim=1)

                # Ensure sequence length is reasonable
                current_seq_len = indices_sequence.size(1)
                if current_seq_len > max_seq_len:
                     # print(f"Warning: Sequence length {current_seq_len} exceeds max_seq_len {max_seq_len}. Truncating.")
                     indices_sequence = indices_sequence[:, :max_seq_len]
                elif current_seq_len < 2: # Need at least 2 tokens for input/target pair
                    print(f"Warning: Sequence length {current_seq_len} is too short (< 2). Skipping batch.")
                    continue


            # 4. Prepare input and target sequences for Transformer
            # Input: indices_sequence[:, :-1] (all tokens except the last)
            # Target: indices_sequence[:, 1:] (all tokens except the first)
            input_seq = indices_sequence[:, :-1]
            target_seq = indices_sequence[:, 1:]

            if input_seq.size(1) == 0: # Check if sequence became too short after slicing
                print(f"Warning: Input sequence length is 0 after slicing. Skipping batch.")
                continue


            # 5. Forward pass through Transformer
            optimizer.zero_grad()
            try:
                logits = transformer_model(input_seq) # Shape: (B, Seq_Len-1, Num_Embeddings)
            except Exception as e:
                 print(f"Error during Transformer forward pass: {e}")
                 print(f"Input sequence shape: {input_seq.shape}")
                 continue


            # 6. Calculate Loss
            # Reshape logits and targets for CrossEntropyLoss
            # Logits: (B * (Seq_Len-1), Num_Embeddings)
            # Target: (B * (Seq_Len-1))
            try:
                loss = criterion(logits.reshape(-1, logits.size(-1)), target_seq.reshape(-1))
            except Exception as e:
                print(f"Error calculating loss: {e}")
                print(f"Logits shape: {logits.shape}, Target shape: {target_seq.shape}")
                continue


            # 7. Backward pass and optimization
            loss.backward()
            torch.nn.utils.clip_grad_norm_(transformer_model.parameters(), 1.0) # Gradient clipping
            optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            writer.add_scalar("Transformer Loss/train", loss.item(), global_step)
            global_step += 1

        # Avoid division by zero if dataloader is empty or all batches were skipped
        if len(dataloader) > 0 and len(loop) > 0 :
             avg_loss = total_loss / len(loop) # Use len(loop) as it reflects processed batches
             print(f"Epoch {epoch+1}/{epochs}, Average Transformer Loss: {avg_loss:.4f}")
             writer.add_scalar("Transformer Loss/epoch", avg_loss, epoch)
        else:
             print(f"Epoch {epoch+1}/{epochs}, No batches processed.")


    print("Transformer training finished.")
    # Return the shapes captured from the first batch
    if quantized_shapes_epoch is None:
        print("Warning: Could not capture quantized shapes during training.")
    return transformer_model, quantized_shapes_epoch

# --- Reconstruction Function ---
@torch.no_grad()
def calculate_reconstruction(transformer_model, vqvae_model, quantized_shapes, num_samples, max_gen_len, device, start_token_idx=0):
    """
    Generates samples using the transformer and reconstructs images using the VQ-VAE decoder.
    Calculates reconstruction metrics (L1, SSIM).

    Args:
        transformer_model: Trained WaveletTransformer.
        vqvae_model: Pre-trained WaveletVQVAE.
        quantized_shapes: List of shapes [(B, H_i, W_i), ...] for each subband's indices map.
                           (Note: B here is dummy, only H, W are needed).
        num_samples (int): Number of samples to generate and reconstruct.
        max_gen_len (int): The total length of the concatenated index sequence to generate.
        device: Torch device ('cuda' or 'cpu').
        start_token_idx (int): Index to use as the starting token for generation.
    """
    if quantized_shapes is None:
        print("Error in calculate_reconstruction: quantized_shapes is None. Cannot proceed.")
        return

    transformer_model.eval()
    vqvae_model.eval()
    transformer_model.to(device)
    vqvae_model.to(device)

    print(f"\nGenerating {num_samples} samples and calculating reconstruction loss...")

    # Prepare starting sequence (e.g., a single start token)
    # Ensure start_token_idx is within the valid range (0 to num_embeddings-1)
    valid_start_token = max(0, min(start_token_idx, transformer_model.num_embeddings - 1))
    start_indices = torch.full((num_samples, 1), valid_start_token, dtype=torch.long, device=device)

    # Generate full index sequences using the transformer
    generated_indices_flat = transformer_model.generate(start_indices, max_gen_len, temperature=0.7) # (B, max_gen_len)

    # --- Decode generated indices using VQ-VAE ---
    reconstructed_coeffs = []
    current_idx = 0
    # Get codebook embeddings from VQ-VAE quantizers
    try:
        codebooks = [q.embeddings.weight.data for q in vqvae_model.quantizers] # List of (num_embed, embed_dim)
    except AttributeError:
        print("Error accessing VQ-VAE quantizer embeddings. Check vqvae_model structure.")
        return


    for i in range(len(quantized_shapes)): # Iterate through LL, LH, HL, HH
        # Get the shape (H, W) for the current subband's index map
        # Use shape from the first batch element as reference: quantized_shapes[i][1:]
        try:
            if len(quantized_shapes[i]) < 3:
                 print(f"Error: Invalid shape format in quantized_shapes at index {i}: {quantized_shapes[i]}")
                 continue
            h, w = quantized_shapes[i][1], quantized_shapes[i][2]
            num_indices_level = h * w
        except IndexError:
            print(f"Error accessing shape dimensions in quantized_shapes at index {i}: {quantized_shapes[i]}")
            continue


        # Extract the portion of generated indices for this level
        # Ensure we don't index out of bounds
        end_idx = current_idx + num_indices_level
        if end_idx > generated_indices_flat.size(1):
            print(f"Warning: Not enough generated indices ({generated_indices_flat.size(1)}) to reconstruct level {i} (needed {end_idx}). Skipping level.")
            # Optionally pad or handle differently
            continue # Skip this level if not enough indices


        indices_level_flat = generated_indices_flat[:, current_idx : end_idx]
        current_idx = end_idx # Update current index position

        # Check if extracted indices match expected number
        if indices_level_flat.size(1) != num_indices_level:
             print(f"Warning: Mismatch in expected ({num_indices_level}) vs extracted ({indices_level_flat.size(1)}) indices for level {i}. Reshaping might fail.")
             # Attempt to pad or truncate if necessary, or skip
             # Padding example (if fewer indices generated):
             if indices_level_flat.size(1) < num_indices_level:
                 padding_size = num_indices_level - indices_level_flat.size(1)
                 # Pad with a default index, e.g., 0
                 padding = torch.zeros((num_samples, padding_size), dtype=torch.long, device=device)
                 indices_level_flat = torch.cat([indices_level_flat, padding], dim=1)
             # Truncating example (if more indices generated - less likely here)
             # indices_level_flat = indices_level_flat[:, :num_indices_level]


        # Reshape indices back to (B, H, W)
        try:
            indices_level = indices_level_flat.view(num_samples, h, w)
        except RuntimeError as e:
            print(f"Error reshaping indices for level {i}: {e}")
            print(f"Target shape: ({num_samples}, {h}, {w}), Flat shape: {indices_level_flat.shape}")
            continue # Skip this level

        # Map indices to embeddings
        # Ensure indices are within the valid range for the codebook
        indices_level = torch.clamp(indices_level, 0, codebooks[i].size(0) - 1)
        try:
            # Shape: (B, H, W, embedding_dim)
            quantized_level = F.embedding(indices_level, codebooks[i])
        except IndexError as e:
             print(f"Error during embedding lookup for level {i}: {e}")
             print(f"Max index found: {indices_level.max()}, Codebook size: {codebooks[i].size(0)}")
             continue # Skip this level


        # Permute to (B, embedding_dim, H, W) for the decoder
        quantized_level = quantized_level.permute(0, 3, 1, 2).contiguous()

        # Decode using the corresponding VQ-VAE decoder
        try:
            recon_coeff = vqvae_model.decoders[i](quantized_level)
            reconstructed_coeffs.append(recon_coeff)
        except Exception as e:
             print(f"Error during VQ-VAE decoding for level {i}: {e}")
             continue # Skip appending this coefficient


    # Check if we have the correct number of coefficients for reconstruction
    if len(reconstructed_coeffs) != vqvae_model.num_levels: # num_levels should be 4 (LL, LH, HL, HH)
        print(f"Error: Expected {vqvae_model.num_levels} reconstructed coefficients, but got {len(reconstructed_coeffs)}. Cannot perform inverse wavelet transform.")
        return

    # Reconstruct the final image using inverse wavelet transform
    try:
        LL, LH, HL, HH = reconstructed_coeffs
        rc = (LL, (LH, HL, HH))
        reconstructed_images = ptwt.waverec2(rc, wavelet) # Shape: (B, C, H_orig, W_orig)
        print("Sample generation and decoding complete.")

        # --- Optional: Compare with original images if available ---
        # You would need a validation dataloader here
        # val_dataloader = DataLoader(...) # Define your validation loader
        # try:
        #     original_batch = next(iter(val_dataloader))
        #     original_images = original_batch['image'][:num_samples].to(device)
        #     # Ensure shapes match for comparison
        #     if reconstructed_images.shape == original_images.shape:
        #         l1_loss = F.l1_loss(reconstructed_images, original_images)
        #         # Ensure data range is appropriate for SSIM (e.g., [0, 1] or [-1, 1])
        #         # Assuming images are [-1, 1] from normalization
        #         ssim_score = ssim(reconstructed_images, original_images)
        #         print(f"Reconstruction L1 Loss (vs validation): {l1_loss.item():.4f}")
        #         print(f"Reconstruction SSIM (vs validation): {ssim_score.item():.4f}")
        #     else:
        #         print("Shape mismatch between reconstructed and original images. Cannot calculate metrics.")
        #         print(f"Reconstructed shape: {reconstructed_images.shape}, Original shape: {original_images.shape}")
        # except StopIteration:
        #     print("Validation dataloader is empty. Cannot compare with original images.")


        # Save generated images (unnormalize first)
        import torchvision.utils as vutils
        save_path = 'generated_samples.png'
        vutils.save_image(reconstructed_images.clamp(-1, 1).add(1).mul(0.5), save_path, nrow=int(math.sqrt(num_samples)))
        print(f"Generated samples saved to {save_path}")

    except ValueError as e:
         print(f"Error during inverse wavelet transform or saving: {e}")
         print("Check coefficient shapes and compatibility.")
    except Exception as e:
        print(f"An unexpected error occurred during reconstruction finalization: {e}")



if __name__ == "__main__":
    # --- Hyperparameters ---
    # Data
    batch_size = 16 # Adjust based on GPU memory
    image_size = 128 # IMPORTANT: Make sure this matches VQVAE training
    num_workers = 0 # Set to 0 to avoid multiprocessing issues initially, increase later if needed

    # VQ-VAE params (MUST match the loaded model)
    vqvae_num_embeddings = 256
    vqvae_embedding_dim = 64
    vqvae_num_levels = 4 # LL, LH, HL, HH (Should match the loaded VQVAE model structure)
    vqvae_model_path = "paths/wavelet_vqvae-original.pth" # Path to your trained VQVAE model

    # Transformer params
    transformer_epochs = 10
    transformer_lr = 1e-4
    transformer_heads = 8
    transformer_layers = 6
    transformer_hidden_dim = 2048 # Feedforward dim in TransformerEncoderLayer
    transformer_dropout = 0.1
    # max_seq_len will be determined dynamically

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- Data Loading ---
    # Initialize the global transform used by the apply_transform function
    global_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)), # Ensure consistent size
            transforms.ToTensor(), # Converts PIL image to a tensor in [0, 1]
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)), # Normalize to [-1, 1]
        ]
    )

    # Using FFHQ dataset as an example
    dataset_name = "merkol/ffhq-256"
    try:
        # Load only a small part for faster testing/debugging
        # dset = load_dataset(dataset_name, split="train[:1%]") # Use 1% of data
        dset = load_dataset(dataset_name, split="train") # Use full dataset for actual training
        print(f"Dataset '{dataset_name}' loaded successfully.")
    except Exception as e:
        print(f"Could not load dataset '{dataset_name}'. Error: {e}")
        print("Please ensure the dataset is available or replace with another.")
        exit()

    # Use the named function for the transform
    dset.set_transform(apply_transform)

    # Limit dataset size for faster example run (optional, remove for full training)
    # dset = dset.select(range(1000))

    dataloader = DataLoader(
        dset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers, # Use 0 first to ensure no pickling errors
        pin_memory=True if device == "cuda" else False,
        drop_last=True # Drop last incomplete batch
    )
    print(f"DataLoader created with batch_size={batch_size}, num_workers={num_workers}")


    # --- Model Loading and Initialization ---

    # Load pre-trained VQ-VAE
    print(f"Loading pre-trained VQ-VAE from {vqvae_model_path}...")
    # Ensure the VQVAE class matches the saved state dict structure
    vqvae_model = WaveletVQVAE(
        num_embeddings=vqvae_num_embeddings,
        embedding_dim=vqvae_embedding_dim,
        num_levels=vqvae_num_levels, # Crucial: Must match the definition in vqvae.py used for training
    )
    try:
        vqvae_model.load_state_dict(torch.load(vqvae_model_path, map_location=device))
        print("VQ-VAE model loaded successfully.")
    except FileNotFoundError:
        print(f"Error: VQ-VAE model file not found at {vqvae_model_path}. Please train VQ-VAE first or check path.")
        exit()
    except RuntimeError as e:
         print(f"Error loading VQ-VAE state dict: {e}")
         print("This often means the WaveletVQVAE class definition in vqvae.py does not match the saved model structure.")
         print(f"Ensure num_levels={vqvae_num_levels} and other parameters match the saved model.")
         exit()
    except Exception as e:
        print(f"An unexpected error occurred loading VQ-VAE state dict: {e}")
        exit()

    vqvae_model.eval() # Set to evaluation mode
    vqvae_model.to(device) # Move VQVAE model to device

    # Determine max_seq_len by processing one batch
    print("Determining sequence length...")
    max_seq_len = 0
    quantized_shapes_ref = None
    try:
        # Get a sample batch
        sample_batch = next(iter(dataloader))
        # Check if data is a dictionary and has the 'image' key
        if isinstance(sample_batch, dict) and 'image' in sample_batch:
            sample_images = sample_batch["image"].to(device)
        elif isinstance(sample_batch, torch.Tensor): # Handle cases where DataLoader might return only tensors
            sample_images = sample_batch.to(device)
        else:
             raise TypeError(f"Unexpected data format from DataLoader: {type(sample_batch)}")

        # Ensure images have 3 channels
        if sample_images.shape[1] == 1:
             sample_images = sample_images.repeat(1, 3, 1, 1)


        with torch.no_grad():
            wavelet_coeffs = get_wavelet_coeffs(sample_images)
            total_indices_length = 0
            quantized_shapes_ref = [] # Store reference shapes
            for i in range(len(wavelet_coeffs)):
                 encoded = vqvae_model.encoders[i](wavelet_coeffs[i])
                 _, _, indices = vqvae_model.quantizers[i](encoded)
                 total_indices_length += indices.shape[1] * indices.shape[2]
                 quantized_shapes_ref.append(indices.shape) # Store (B, H_i, W_i)
            max_seq_len = total_indices_length
            print(f"Determined max sequence length: {max_seq_len}")
            # Add buffer to max_seq_len just in case? Usually not needed if input size is fixed.
            # max_seq_len += 10
    except StopIteration:
         print("Error: DataLoader yielded no batches. Check dataset and batch size.")
         exit()
    except Exception as e:
        print(f"Error determining sequence length: {e}")
        print("Check data loading, transformations, and VQ-VAE forward pass.")
        exit()

    if max_seq_len <= 1:
        print(f"Error: Determined max_seq_len ({max_seq_len}) is too small. Cannot train transformer.")
        exit()


    # Initialize Transformer
    print("Initializing Transformer model...")
    transformer_model = WaveletTransformer(
        num_embeddings=vqvae_num_embeddings,
        embedding_dim=vqvae_embedding_dim,
        num_heads=transformer_heads,
        num_layers=transformer_layers,
        hidden_dim=transformer_hidden_dim,
        max_seq_len=max_seq_len + 1,
        quantized_shapes=quantized_shapes_ref,   # ← here
        dropout=transformer_dropout,
    ).to(device)

    # --- Training Setup ---
    optimizer = torch.optim.AdamW(transformer_model.parameters(), lr=transformer_lr)
    # Use ignore_index=-100 if padding is introduced later, otherwise standard CE is fine
    criterion = nn.CrossEntropyLoss()
    log_dir = "runs/wavelet_transformer"
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard logs will be saved to: {log_dir}")

    # --- Train the Transformer ---
    trained_transformer, final_quantized_shapes = train_transformer(
        transformer_model,
        vqvae_model,
        dataloader,
        optimizer,
        criterion,
        transformer_epochs,
        device,
        writer,
        max_seq_len # Pass the determined max length
    )

    # --- Save the Trained Transformer ---
    transformer_save_path = "wavelet_transformer.pth"
    print(f"Saving trained transformer model to {transformer_save_path}...")
    torch.save(trained_transformer.state_dict(), transformer_save_path)
    print("Model saved.")

    # --- Calculate Reconstruction Loss (using generated samples) ---
    # Use the reference shapes determined before training
    calculate_reconstruction(
        transformer_model=trained_transformer,
        vqvae_model=vqvae_model,
        quantized_shapes=quantized_shapes_ref, # Use shapes determined earlier
        num_samples=16, # Number of samples to generate
        max_gen_len=max_seq_len,
        device=device,
        start_token_idx=0 # Assuming 0 is a valid index
    )

    writer.close()
    print("Finished.")
