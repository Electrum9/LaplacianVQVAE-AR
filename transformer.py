import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

import pywt
import ptwt

from vqvae import WaveletVQVAE       
from rqvae import RQTransformer     

# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Helper funcs for true DWT / IDWT
def build_wavelet_pyramid(x: torch.Tensor,
                          num_levels: int,
                          wavelet_name: str = 'db8'):
    """
    Multi‐level 2D DWT on a batch of images.
    x: [B, C, H, W]
    returns: list [cA_n, (cH_n,cV_n,cD_n), …, (cH_1,cV_1,cD_1)]
    """
    return ptwt.wavedec2(x, pywt.Wavelet(wavelet_name), level=num_levels)

def reconstruct_from_wavelet_pyramid(coeffs: list,
                                     wavelet_name: str = 'db8') -> torch.Tensor:
    """
    True inverse wavelet transform.
    coeffs: output of build_wavelet_pyramid
    returns: [B, C, H, W]
    """
    return ptwt.waverec2(coeffs, pywt.Wavelet(wavelet_name))


# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
def load_vqvae_model(checkpoint_path: str,
                     num_embeddings=1024,
                     embedding_dim=64,
                     num_levels=3) -> WaveletVQVAE:
    """
    Instantiate a WaveletVQVAE and load weights.
    """
    model = WaveletVQVAE(
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        num_levels=num_levels
    )
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get('state_dict', ckpt)
    model.load_state_dict(state)
    model.to(device).eval()
    print(f"Loaded VQ-VAE from {checkpoint_path}")
    return model

def load_transformer_model(checkpoint_path: str,
                           vqvae_model: WaveletVQVAE) -> RQTransformer:
    """
    Instantiate an RQTransformer and load weights.
    """
    num_quantizers = len(vqvae_model.quantizers)
    embedding_dim   = vqvae_model.quantizers[0].embedding_dim
    num_embeddings  = vqvae_model.quantizers[0].num_embeddings

    model = RQTransformer(
        d_model=512,
        spatial_nhead=8,
        depth_nhead=4,
        spatial_layers=6,
        depth_layers=2,
        code_embedding_dim=embedding_dim,
        num_embeddings=num_embeddings,
        num_quantizers=num_quantizers,
        max_seq_len=64
    )
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get('state_dict', ckpt)
    model.load_state_dict(state)
    model.to(device).eval()
    print(f"Loaded Transformer from {checkpoint_path}")
    return model


# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
def visualize_reconstruction(vqvae_model: WaveletVQVAE,
                             x: torch.Tensor,
                             wavelet_name: str = 'db8'):
    """
    Reconstruct a batch of images via wavelet VQ-VAE + true inverse DWT.
    """
    vqvae_model.eval()
    with torch.no_grad():
        x = x.to(device)
        # 1) DWT
        coeffs = build_wavelet_pyramid(x, num_levels=vqvae_model.num_levels-1, wavelet_name=wavelet_name)

        # 2) flatten to list of tensors for VQ-VAE:
        flat = [ coeffs[0] ]
        for (cH, cV, cD) in coeffs[1:]:
            flat.append(torch.cat([cH, cV, cD], dim=1))

        # 3) quantize & decode each band
        recon_flat, _, _ = vqvae_model(flat)

        # 4) unflatten back to nested wavelet structure
        recon_coeffs = [ recon_flat[0] ]
        for lvl in range(1, len(recon_flat)):
            concat = recon_flat[lvl]
            cH, cV, cD = torch.chunk(concat, chunks=3, dim=1)
            recon_coeffs.append((cH, cV, cD))

        # 5) true IDWT
        recon = reconstruct_from_wavelet_pyramid(recon_coeffs, wavelet_name=wavelet_name)

        # denormalize & to CPU
        x_cpu   = ((x + 1) / 2).cpu()
        recon_cpu = ((recon + 1) / 2).cpu()

        # plot
        n = min(4, x_cpu.size(0))
        fig, axs = plt.subplots(2, n, figsize=(n*3, 6))
        for i in range(n):
            axs[0,i].imshow(x_cpu[i].permute(1,2,0))
            axs[0,i].set_title('Original'); axs[0,i].axis('off')
            axs[1,i].imshow(recon_cpu[i].permute(1,2,0))
            axs[1,i].set_title('Reconstructed'); axs[1,i].axis('off')
        plt.tight_layout()
        plt.show()
        return recon_cpu


# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
def generate_images(transformer_model: RQTransformer,
                    vqvae_model: WaveletVQVAE,
                    num_images: int = 4,
                    seq_len: int = 64,
                    wavelet_name: str = 'db8'):
    """
    Autoregressively sample code tokens with the transformer,
    decode via VQ-VAE decoders, then do true inverse DWT.
    """
    transformer_model.eval()
    vqvae_model.eval()
    with torch.no_grad():
        B = num_images
        # embedding dim
        D = vqvae_model.quantizers[0].embedding_dim

        # start with empty code embeddings
        code_emb = torch.zeros(B, 0, D, device=device)
        all_tokens = []

        # sample token streams
        for _ in range(seq_len):
            nxt = transformer_model(code_emb, generation=True)  # [B, 1, depth]
            all_tokens.append(nxt)
            # embed
            emb = torch.zeros(B, D, device=device)
            for d, tok in enumerate(nxt.unbind(-1)):
                emb += vqvae_model.quantizers[d].embeddings(tok)
            code_emb = torch.cat([code_emb, emb.unsqueeze(1)], dim=1)

        # stack tokens [B, seq_len, depth]
        tokens = torch.stack(all_tokens, dim=1)

        # reshape into per-level chunks
        per_level = seq_len // vqvae_model.num_levels
        recon_flat = []
        for lvl in range(vqvae_model.num_levels):
            start = lvl * per_level
            end   = (lvl+1) * per_level
            # get tokens for this level & invert via decoder
            lvl_toks = tokens[:, start:end, :].reshape(B, -1)
            # decode: get embeddings then pass through decoder
            # (your VQ-VAE decoder expects [B, D, H, W] – reshape accordingly)
            # here we assume square H=W=int(sqrt(per_level))
            H = W = int(np.sqrt(per_level))
            quant = torch.zeros(B, D, H, W, device=device)
            for d in range(tokens.size(-1)):
                depth_tok = lvl_toks[:, d::tokens.size(-1)].reshape(B, -1)
                emb = vqvae_model.quantizers[lvl].embeddings(depth_tok)
                emb = emb.view(B, H, W, D).permute(0,3,1,2)
                quant += emb
            recon_flat.append(vqvae_model.decoders[lvl](quant))

        # re‐nest for true IDWT
        recon_coeffs = [ recon_flat[0] ]
        for lvl in range(1, len(recon_flat)):
            cH, cV, cD = torch.chunk(recon_flat[lvl], chunks=3, dim=1)
            recon_coeffs.append((cH, cV, cD))

        # invert
        imgs = reconstruct_from_wavelet_pyramid(recon_coeffs, wavelet_name=wavelet_name)
        imgs = ((imgs + 1)/2).cpu()

        # visualize grid
        fig, axs = plt.subplots(1, num_images, figsize=(num_images*3, 3))
        for i in range(num_images):
            axs[i].imshow(imgs[i].permute(1,2,0).numpy())
            axs[i].axis('off')
        plt.tight_layout()
        plt.show()

        return imgs


# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
if __name__ == "__main__":
    # 1) load models
    vqvae_ckpt = "checkpoints/wavelet_vqvae-l2.pth"
    transformer_ckpt = "checkpoints/transformer-wavelet.pth"

    vqvae = load_vqvae_model(vqvae_ckpt,
                             num_embeddings=1024,
                             embedding_dim=64,
                             num_levels=3)
    transformer = load_transformer_model(transformer_ckpt, vqvae)

    # 2) get a test batch
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize((0.5,)*3, (0.5,)*3)
    ])
    ds = datasets.CIFAR10(root="data", train=False, download=True, transform=transform)
    loader = DataLoader(ds, batch_size=4, shuffle=True)
    imgs, _ = next(iter(loader))

    # 3) visualize reconstructions
    print("Reconstruction:")
    visualize_reconstruction(vqvae, imgs)

    # 4) sample new images
    print("Sampling:")
    _ = generate_images(transformer, vqvae, num_images=4, seq_len=64)
    print("Done!")