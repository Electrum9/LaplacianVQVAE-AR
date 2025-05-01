# run_inference.py

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import ptwt
import pywt
from PIL import Image

from vqvae import WaveletVQVAE, get_wavelet_coeffs
from transformer import WaveletTransformer  # adjust import path if needed

# --- Configuration ---
device = "cuda" if torch.cuda.is_available() else "cpu"
vqvae_path = "paths/wavelet_vqvae-original.pth"
transformer_path = "wavelet_transformer.pth"
reference_image_path = "generated_samples_0.png"  # your input image
output_figure_path = "wavelet_and_recon.png"
image_size = 128
num_heads = 8
num_layers = 6
hidden_dim = 2048
dropout = 0.1

# --- 1) Load VQ-VAE and infer embedding dims ---
state = torch.load(vqvae_path, map_location=device)
# find first quantizer embedding weight key
key = next(k for k in state if k.endswith("quantizers.0.embeddings.weight"))
num_embeddings, embedding_dim = state[key].shape

vqvae = WaveletVQVAE(
    num_embeddings=num_embeddings,
    embedding_dim=embedding_dim,
    num_levels=4
)
vqvae.load_state_dict(state)
vqvae.to(device).eval()

# --- 2) Compute quantized_shapes from a reference image ---
ref_img = Image.open(reference_image_path).convert("RGB")
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5)
])
x = transform(ref_img).unsqueeze(0).to(device)  # (1,3,H,W)

with torch.no_grad():
    coeffs = get_wavelet_coeffs(x)  # list of 4 subbands
    quantized_shapes = []
    for i, sub in enumerate(coeffs):
        e = vqvae.encoders[i](sub)
        _, _, idxs = vqvae.quantizers[i](e)
        B, H, W = idxs.shape
        quantized_shapes.append((B, H, W))

# compute max_seq_len
max_seq_len = sum(h * w for (_, h, w) in quantized_shapes)

# --- 3) Load Transformer ---
transformer = WaveletTransformer(
    num_embeddings=num_embeddings,
    embedding_dim=embedding_dim,
    num_heads=num_heads,
    num_layers=num_layers,
    hidden_dim=hidden_dim,
    max_seq_len=max_seq_len + 1,
    quantized_shapes=quantized_shapes,
    dropout=dropout
)
transformer.load_state_dict(torch.load(transformer_path, map_location=device))
transformer.to(device).eval()

# --- 4) Generate token sequence ---
with torch.no_grad():
    start = torch.zeros(1, 1, dtype=torch.long, device=device)
    gen_seq = transformer.generate(start, max_len=max_seq_len, temperature=1.0)  # (1, max_seq_len)

# --- 5) Decode subbands to coefficient maps ---
codebooks = [q.embeddings.weight.detach() for q in vqvae.quantizers]
subbands = []
ptr = 0
for (B, H, W), codebook, decoder in zip(quantized_shapes, codebooks, vqvae.decoders):
    n = H * W
    flat = gen_seq[:, ptr : ptr + n].clamp(0, codebook.size(0) - 1)
    ptr += n

    emb = F.embedding(flat, codebook)            # (1, n, D)
    emb = emb.view(1, H, W, -1).permute(0, 3, 1, 2)  # (1, D, H, W)

    with torch.no_grad():
        coeff = decoder(emb).squeeze(0)           # (C_out, H_out, W_out)
    # average across channels to get a single map
    coeff_map = coeff.mean(0).cpu().numpy()        # (H_out, W_out)
    subbands.append(coeff_map)

LL, LH, HL, HH = subbands

# --- 6) Inverse wavelet to reconstruct final image ---
# each subband is (H, W), wrap back to (1, H, W)
LL_t = torch.tensor(LL).unsqueeze(0)
LH_t = torch.tensor(LH).unsqueeze(0)
HL_t = torch.tensor(HL).unsqueeze(0)
HH_t = torch.tensor(HH).unsqueeze(0)
recon = ptwt.waverec2((LL_t, (LH_t, HL_t, HH_t)), pywt.Wavelet("db8"))
recon = recon.squeeze(0).cpu().numpy()  # (H, W)
# normalize to [0,1]
recon = (recon - recon.min()) / (recon.max() - recon.min())

# --- 7) Plot and save all five images side-by-side ---
fig, axes = plt.subplots(1, 5, figsize=(20, 4))
titles = ["LL", "LH", "HL", "HH", "Reconstruction"]
images = [LL, LH, HL, HH, recon]
for ax, img, title in zip(axes, images, titles):
    ax.imshow(img, cmap="gray")
    ax.set_title(title)
    ax.axis("off")

plt.tight_layout()
plt.savefig(output_figure_path, dpi=150)
print(f"Saved ➜ {output_figure_path}")
plt.show()
