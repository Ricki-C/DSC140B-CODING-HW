import numpy as np
import matplotlib.pyplot as plt
import torch


def plot_parking(time, temp, label, ax=None):
    """Scatter plot of the parking availability dataset."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    found = label == 1

    ax.scatter(
        time[found], temp[found],
        c="#5ec55e", s=45, alpha=0.8, edgecolors="none", label="Found Parking",
    )
    ax.scatter(
        time[~found], temp[~found],
        c="#e06060", s=45, alpha=0.8, edgecolors="none", label="No Parking",
    )

    ax.set_xlabel(r"$x_1$ = time of day", fontsize=14)
    ax.set_ylabel(r"$x_2$ = temperature", fontsize=14)
    ax.legend(fontsize=12, loc="upper left")
    ax.set_xticks([])
    ax.set_yticks([])

    return ax


def plot_decision_boundary(model, time, temp, label, mu, sigma, ax=None):
    """Plot the model's decision boundary with the data overlaid."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    found = label == 1

    # Build a grid in original (unstandardized) space
    t1 = np.linspace(time.min() - 1, time.max() + 1, 200)
    t2 = np.linspace(temp.min() - 1, temp.max() + 1, 200)
    g1, g2 = np.meshgrid(t1, t2)

    # Standardize the grid with the same mu/sigma used for training
    grid = np.column_stack([g1.ravel(), g2.ravel()])
    grid_std = (grid - mu) / sigma
    grid_tensor = torch.tensor(grid_std, dtype=torch.float32)

    # Predict
    with torch.no_grad():
        probs = model(grid_tensor).numpy().reshape(g1.shape)

    ax.contourf(g1, g2, probs, levels=[0, 0.5, 1], colors=["#f5c4c4", "#c4f5c4"], alpha=0.5)
    ax.contour(g1, g2, probs, levels=[0.5], colors="k", linewidths=1)
    plot_parking(time, temp, label, ax=ax)

    return ax


def plot_feature_map(X_original, X_transformed, label):
    """Side-by-side plot of data in original space vs. learned feature space."""
    found = label == 1

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Original space
    ax = axes[0]
    ax.scatter(X_original[found, 0], X_original[found, 1], c="#5ec55e", s=45, alpha=0.8, edgecolors="none", label="Found Parking")
    ax.scatter(X_original[~found, 0], X_original[~found, 1], c="#e06060", s=45, alpha=0.8, edgecolors="none", label="No Parking")
    ax.set_title("Original space", fontsize=14)
    ax.set_xlabel(r"$x_1$", fontsize=13)
    ax.set_ylabel(r"$x_2$", fontsize=13)
    ax.legend(fontsize=10)

    # Feature space
    ax = axes[1]
    ax.scatter(X_transformed[found, 0], X_transformed[found, 1], c="#5ec55e", s=45, alpha=0.8, edgecolors="none", label="Found Parking")
    ax.scatter(X_transformed[~found, 0], X_transformed[~found, 1], c="#e06060", s=45, alpha=0.8, edgecolors="none", label="No Parking")
    ax.set_title(r"Learned feature space  $\varphi(\vec{x})$", fontsize=14)
    ax.set_xlabel(r"$z_1$", fontsize=13)
    ax.set_ylabel(r"$z_2$", fontsize=13)
    ax.legend(fontsize=10)

    plt.tight_layout()
    return axes
