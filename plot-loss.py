import sys
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set a nice Seaborn style
sns.set(style="whitegrid", context="talk")

# Path to your text file with loss values (each value on a new line)
loss_file_path = Path(sys.argv[1])

# Load the loss values
with open(loss_file_path, "r") as f:
    losses = [float(line.strip()) for line in f if line.strip()]

# Generate the x-axis (epochs or iterations)
steps = np.arange(1, len(losses) + 1)

# Create the plot
plt.figure(figsize=(10, 6))
sns.lineplot(x=steps, y=losses, linewidth=2.5, color="#007acc", label="Loss")

# Enhance the aesthetics
plt.title("Training Loss Over Time", fontsize=18, weight='bold')
plt.xlabel("Step", fontsize=14)
plt.ylabel("Loss", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12)
plt.tight_layout()
sns.despine()

# Show the plot
plt.savefig(loss_file_path.name + "-plot.png")

