import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm
import pywt
import ptwt
from datasets import load_dataset

import torchmetrics

from torch.utils.tensorboard import SummaryWriter

ssim = torchmetrics.StructuralSimilarityIndexMeasure().to('cuda')
wavelet = pywt.Wavelet("db8")


def get_wavelet_coeffs(img_tensor):

    # 2D DWT (1 level)
    coeffs = ptwt.wavedec2(img_tensor, wavelet, level=1)
    LL, (LH, HL, HH) = coeffs

    return [LL, LH, HL, HH]


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
        self.embeddings.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)

    def forward(self, inputs):
        # Convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self.embedding_dim)

        # Calculate distances
        distances = (
            torch.sum(flat_input**2, dim=1, keepdim=True)
            + torch.sum(self.embeddings.weight**2, dim=1)
            - 2 * torch.matmul(flat_input, self.embeddings.weight.t())
        )

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(
            encoding_indices.shape[0], self.num_embeddings, device=inputs.device
        )
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
        return (
            quantized.permute(0, 3, 1, 2).contiguous(),
            loss,
            encoding_indices.view(input_shape[:-1]),
        )


class WaveletVQVAE(nn.Module):
    """
    VQ-VAE for Wavelet coefficients
    """

    def __init__(self, num_embeddings=1024, embedding_dim=64, num_levels=4):
        super().__init__()
        self.num_levels = num_levels

        # Encoders for each level of the Laplacian pyramid
        self.encoders = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
                    nn.ReLU(True),
                    nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
                    nn.ReLU(True),
                    nn.Conv2d(128, embedding_dim, kernel_size=3, stride=1, padding=1),
                )
                for _ in range(num_levels)
            ]
        )

        # Vector Quantizers for each set of wavelet coeffs
        self.quantizers = nn.ModuleList(
            [VectorQuantizer(num_embeddings, embedding_dim) for _ in range(num_levels)]
        )

        # Decoders for each level
        self.decoders = nn.ModuleList(
            [
                nn.Sequential(
                    nn.ConvTranspose2d(
                        embedding_dim,
                        128,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        output_padding=1,
                    ),  # 66 -> 135
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(
                        128, 64, kernel_size=4, stride=2, padding=1, output_padding=1
                    ),  # 66 -> 135
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),  # 135 -> 135
                )
                for _ in range(num_levels)
            ]
        )

    def forward(self, wavelet_coeffs):
        """
        Forward pass through the VQ-VAE for each set of Wavelet coeffs

        Args:
            wavelet_coeffs: Set of wavelet coefficients (LL, LH, HL, HH)

        Returns:
            reconstructed_levels: Reconstructed pyramid levels
            total_loss: Quantization loss
            indices_list: List of quantized indices for each level
        """
        assert (
            len(wavelet_coeffs) == self.num_levels
        ), "Number of sets of wavelet coeffs doesn't match"

        reconstructed_coeffs = []
        total_loss = 0
        all_indices = []

        for i in range(self.num_levels):
            # Encode
            encoded = self.encoders[i](wavelet_coeffs[i])

            # Quantize
            quantized, loss, indices = self.quantizers[i](encoded)
            total_loss += loss
            all_indices.append(indices)

            # Decode
            reconstructed = self.decoders[i](quantized)
            reconstructed_coeffs.append(reconstructed)

        return reconstructed_coeffs, total_loss, all_indices


# Training function
def train_wavelet_vqvae(model, dataloader, optimizer, epochs=10, device="cuda"):
    """
    Train the Wavelet VQ-VAE model
    """
    writer = SummaryWriter(log_dir="runs/vqvae-l1-no-overall")
    model.to(device)
    model.train()
    print("Training Wavelet VQ-VAE...")
    for epoch in range(epochs):
        total_loss = 0
        loop = tqdm(dataloader, desc=f"Epoch {epoch}", leave=False)

        for batch_idx, data in enumerate(
            loop
        ):  # Unpack properly - datasets return (data, label)
            data = data["image"].to(device)

            # Build Wavelet pyramid
            wavelet_coeffs = get_wavelet_coeffs(data)

            # Forward pass
            reconstructed_wavelet_coeffs, vq_loss, indices = model(wavelet_coeffs)

            # Compute reconstruction loss
            recon_loss = 0
            for i in range(len(wavelet_coeffs)):
                recon_loss += F.l1_loss(
                    reconstructed_wavelet_coeffs[i], wavelet_coeffs[i]
                )
                # L1 loss used because of sparse outputs for wavelet coeffs (penalize small errors more)

            # Total loss
            loss = recon_loss + vq_loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            loop.set_postfix(loss=loss.item())
            writer.add_scalar("Loss/train", loss.item(), batch_idx)

            if batch_idx % 100 == 0:
                print(f"Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}")

            if batch_idx % 200 == 0:
                for i in range(len(wavelet_coeffs)):
                    writer.add_image(
                        f"Reconstructed Wavelet Coeffs {i}",
                        reconstructed_wavelet_coeffs[i][0],
                        batch_idx,
                    )
                    writer.add_image(
                        f"Wavelet Coeffs {i}", wavelet_coeffs[i][0], batch_idx
                    )

        print(f"Epoch: {epoch}, Average Loss: {total_loss/len(dataloader):.4f}")

    return model


if __name__ == "__main__":

    # Hyperparameters
    batch_size = 32
    vqvae_epochs = 15
    transformer_epochs = 10
    learning_rate = 3e-4
    num_embeddings = 256
    embedding_dim = 64
    num_quantizers = 4
    num_levels = 3
    temperature = 0.5  # Temperature for soft sampling
    device = "cuda"

    transform = transforms.Compose(
        [
            transforms.ToTensor(),  # Converts PIL image to a tensor in [0, 1]
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )
    # This loads the dataset (you can specify 'train', 'validation', etc., if available)
    dset = load_dataset("merkol/ffhq-256", split="train")
    dset.set_transform(
        lambda examples: {
            "image": torch.stack([transform(img) for img in examples["image"]])
        }
    )
    dataloader = DataLoader(dset, batch_size=batch_size, shuffle=True)

    # Create improved models
    vqvae_model = WaveletVQVAE(
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        num_levels=4,
    )

    # Optimizers
    vqvae_optimizer = torch.optim.Adam(vqvae_model.parameters(), lr=learning_rate)

    # Train improved VQ-VAE first
    print("Training Improved Laplacian VQ-VAE...")
    vqvae_model = train_wavelet_vqvae(
        vqvae_model, dataloader, vqvae_optimizer, epochs=vqvae_epochs, device=device
    )
    torch.save(vqvae_model.state_dict(), "wavelet_vqvae.pth")
