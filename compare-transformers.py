#!/usr/bin/env python3
import time
import csv
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from datasets import load_dataset
from fvcore.nn import FlopCountAnalysis  # for FLOP counting

from standard_transformer import WaveletTransformer as StandardTransformer
from transformer import WaveletTransformer as ARTransformer
from vqvae import WaveletVQVAE, get_wavelet_coeffs
from standard_transformer import apply_transform 

import standard_transformer
import transformer

from torch.cuda.amp import autocast, GradScaler
torch.backends.cudnn.benchmark = True

def train_model(model, vqvae, loader, optimizer, criterion, epochs, device):
    model.to(device)
    vqvae.to(device).eval()
    scaler = GradScaler() if device.startswith("cuda") else None
    times, losses = [], []
    for epoch in range(1, epochs+1):
        model.train()
        start = time.time()
        total_loss = 0.0
        for batch in tqdm(loader, desc=f"{model.__class__.__name__} Epoch {epoch}/{epochs}"):
            imgs = batch["image"].to(device)
            with torch.no_grad():
                coeffs = get_wavelet_coeffs(imgs)
                idxs = []
                for i, c in enumerate(coeffs):
                    enc = vqvae.encoders[i](c)
                    _, _, ix = vqvae.quantizers[i](enc)
                    idxs.append(ix.reshape(imgs.size(0), -1))
                seq = torch.cat(idxs, dim=1)
                inp, tgt = seq[:, :-1], seq[:, 1:]

            optimizer.zero_grad()
            if scaler:
                with autocast():
                    logits = model(inp)
                    loss = criterion(
                        logits.reshape(-1, logits.size(-1)),
                        tgt.reshape(-1)
                    )
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(inp)
                loss = criterion(
                    logits.reshape(-1, logits.size(-1)),
                    tgt.reshape(-1)
                )
                loss.backward()
                optimizer.step()

            total_loss += loss.item()

        if device.startswith("cuda"):
            torch.cuda.synchronize()
        elapsed = time.time() - start
        avg_loss = total_loss / len(loader)
        times.append(elapsed)
        losses.append(avg_loss)
        print(f"{model.__class__.__name__} Epoch {epoch}: time={elapsed:.2f}s  loss={avg_loss:.4f}")

    return times, losses

def main():
    # Config
    BATCH_SIZE = 16
    IMG_SIZE   = 128
    WORKERS    = 0
    EPOCHS     = 5
    DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

    # Data setup
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,)*3, (0.5,)*3),
    ])
    standard_transformer.global_transform = transform
    transformer.global_transform = transform

    dset = load_dataset("merkol/ffhq-256", split="train")
    dset.set_transform(apply_transform)
    loader = DataLoader(dset, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=WORKERS, pin_memory=(DEVICE=="cuda"),
                        drop_last=True)

    # VQ-VAE
    vqvae = WaveletVQVAE(num_embeddings=256, embedding_dim=64, num_levels=4)
    vqvae.load_state_dict(torch.load("paths/wavelet_vqvae-original.pth", map_location=DEVICE))
    vqvae.eval().to(DEVICE)

    # Compute shapes and seq length
    sample = next(iter(loader))
    imgs = sample["image"].to(DEVICE)
    quantized_shapes = []
    full_len = 0
    with torch.no_grad():
        coeffs = get_wavelet_coeffs(imgs)
        for i, c in enumerate(coeffs):
            _, _, idx = vqvae.quantizers[i](vqvae.encoders[i](c))
            B, H, W = idx.shape
            quantized_shapes.append((B, H, W))
            full_len += H * W

    # Give the standard transformer a +1 buffer, AR transformer exact length
    std_seq_len = full_len + 1
    ar_seq_len  = full_len

    std_model = StandardTransformer(
        num_embeddings=256,
        embedding_dim=64,
        num_heads=8,
        num_layers=6,
        hidden_dim=2048,
        max_seq_len=std_seq_len,     # one-token buffer for standard
        dropout=0.1
    ).to(DEVICE)

    ar_model = ARTransformer(
        num_embeddings=256,
        embedding_dim=64,
        num_heads=8,
        num_layers=6,
        hidden_dim=2048,
        max_seq_len=ar_seq_len,      # exact full_len for AR path
        quantized_shapes=quantized_shapes,
        dropout=0.1
    ).to(DEVICE)

    # FLOP counting
    dummy_std = torch.randint(0, 256, (1, std_seq_len), device=DEVICE)
    dummy_ar  = torch.randint(0, 256, (1, ar_seq_len),  device=DEVICE)
    flops_std = FlopCountAnalysis(std_model, dummy_std).total() / 1e9
    flops_ar  = FlopCountAnalysis(ar_model,  dummy_ar).total()  / 1e9
    print(f"Standard GFLOPs: {flops_std:.2f}, Wavelet-AR GFLOPs: {flops_ar:.2f}")

    # Plot FLOPs bar chart
    plt.figure()
    plt.bar(["Standard", "Wavelet-AR"], [flops_std, flops_ar], width=0.6)
    plt.ylabel("GFLOPs per forward")
    plt.title("Compute Cost Comparison")
    plt.grid(True, axis="y")
    plt.savefig("compare_flops.png")

    # Optimizers & Loss
    opt_std = torch.optim.AdamW(std_model.parameters(), lr=1e-4)
    opt_ar  = torch.optim.AdamW(ar_model.parameters(),  lr=1e-4)
    crit    = nn.CrossEntropyLoss()

    # Train & record
    times_std, losses_std = train_model(std_model, vqvae, loader, opt_std, crit, EPOCHS, DEVICE)
    times_ar,  losses_ar  = train_model(ar_model,  vqvae, loader, opt_ar,  crit, EPOCHS, DEVICE)

    # Save CSV
    with open("compare_results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model","epoch","time_s","loss"])
        for e,(t,l) in enumerate(zip(times_std,losses_std),1):
            w.writerow(["standard", e, t, l])
        for e,(t,l) in enumerate(zip(times_ar,losses_ar),1):
            w.writerow(["wavelet_ar", e, t, l])

    # Time plot
    epochs = list(range(1, EPOCHS+1))
    plt.figure(); plt.plot(epochs, times_std, "o-", label="Standard"); plt.plot(epochs, times_ar, "o-", label="Wavelet-AR")
    plt.xlabel("Epoch"); plt.ylabel("Time (s)"); plt.title("Epoch Time Comparison"); plt.legend(); plt.grid(True)
    plt.savefig("compare_time.png")

    # Loss plot
    plt.figure(); plt.plot(epochs, losses_std, "o-", label="Standard"); plt.plot(epochs, losses_ar, "o-", label="Wavelet-AR")
    plt.xlabel("Epoch"); plt.ylabel("Avg Loss"); plt.title("Loss Comparison"); plt.legend(); plt.grid(True)
    plt.savefig("compare_loss.png")

    print("Done! CSV and plots saved.")

if __name__ == "__main__":
    main()
