"""Week 1: fixed undirected consensus, explicit Euler comparison.
Run: python consensus.py
Requires: numpy, matplotlib
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def simulate(L, x0, h=0.01, T=10.0):
    """Return times and states; this example accepts undirected Laplacians."""
    L = np.asarray(L, dtype=float)
    x0 = np.asarray(x0, dtype=float)
    if L.shape != (x0.size, x0.size):
        raise ValueError("L and x0 have incompatible shapes")
    if not np.allclose(L, L.T) or not np.allclose(L.sum(axis=1), 0):
        raise ValueError("Expected a symmetric Laplacian with zero row sums")
    off_diagonal = L - np.diag(np.diag(L))
    if np.any(off_diagonal > 1e-12):
        raise ValueError("Off-diagonal entries must be nonpositive")
    eigenvalues = np.linalg.eigvalsh(L)
    if h <= 0 or T <= 0:
        raise ValueError("h and T must be positive")
    if eigenvalues[-1] > 0 and h >= 2 / eigenvalues[-1]:
        raise ValueError("Euler step is too large for disagreement to decay")
    steps = round(T / h)
    if not np.isclose(steps * h, T):
        raise ValueError("Choose T as an integer multiple of h")
    time = np.arange(steps + 1) * h
    states = np.zeros((steps + 1, x0.size))
    states[0] = x0
    for k in range(steps):
        states[k + 1] = states[k] - h * (L @ states[k])
    return time, states


TOPOLOGIES = {
    "Chain": np.array([[1, -1, 0], [-1, 2, -1], [0, -1, 1]], float),
    "Complete": np.array([[2, -1, -1], [-1, 2, -1], [-1, -1, 2]], float),
    "Isolated agent 3": np.array([[1, -1, 0], [-1, 1, 0], [0, 0, 0]], float),
    "Weighted chain": np.array([[2, -2, 0], [-2, 3, -1], [0, -1, 1]], float),
}


def main():
    x0 = np.array([3, 6, 12], dtype=float)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey=True)
    for ax, (name, L) in zip(axes.flat, TOPOLOGIES.items()):
        time, states = simulate(L, x0)
        print(name)
        print("  eigenvalues:", np.linalg.eigvalsh(L))
        print("  final states:", states[-1])
        print("  max mean drift:", np.max(np.abs(states.mean(axis=1) - x0.mean())))
        for i in range(3):
            ax.plot(time, states[:, i], label=f"Agent {i + 1}")
        ax.axhline(x0.mean(), color="black", linestyle="--", label="Initial global mean")
        ax.set_title(name)
        ax.set_xlabel("Time")
        ax.set_ylabel("State")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    output = Path(__file__).with_name("consensus-comparison.png")
    fig.savefig(output, dpi=160)
    print("Figure saved to:", output)
    plt.show()


if __name__ == "__main__":
    main()
