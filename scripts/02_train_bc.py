import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from reacher_bc.core import BCPolicy, ensure_dir, set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demos", type=str, default="artifacts/demos.npz")
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=str, default="artifacts/bc_policy.pt")
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = np.load(args.demos)
    x = data["observations"].astype(np.float32)
    y = data["actions"].astype(np.float32)

    obs_mean = x.mean(axis=0, keepdims=True)
    obs_std = x.std(axis=0, keepdims=True) + 1e-6
    x_norm = (x - obs_mean) / obs_std

    dataset = TensorDataset(torch.tensor(x_norm), torch.tensor(y))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    policy = BCPolicy(x.shape[1], y.shape[1]).to(device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=3e-4, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    losses = []
    for _ in tqdm(range(args.epochs), desc="Training BC"):
        epoch_losses = []
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = policy(xb)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        losses.append(float(np.mean(epoch_losses)))

    out_path = Path(args.out)
    ensure_dir(out_path.parent)
    torch.save(
        {
            "model_state_dict": policy.state_dict(),
            "obs_mean": obs_mean,
            "obs_std": obs_std,
            "obs_dim": x.shape[1],
            "act_dim": y.shape[1],
            "losses": losses,
        },
        out_path,
    )

    plt.figure(figsize=(7, 4))
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("Behavior Cloning Training Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = out_path.parent / "training_loss.png"
    plt.savefig(plot_path, dpi=160)

    print(f"device: {device}")
    print(f"saved_model: {out_path}")
    print(f"saved_plot: {plot_path}")
    print(f"final_train_mse: {losses[-1]:.6f}")


if __name__ == "__main__":
    main()

