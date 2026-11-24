import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    "axes.linewidth": 1.2,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
})

# Domain: 0 <= omega*t <= 2*pi at z = 0
wt = np.linspace(0, 2 * np.pi, 1000)
propagator = np.exp(-1j * wt)

cases = [
    {
        "panel": "(a)",
        "A_H": 1.0,          # Horizontal amplitude (i_hat component)
        "A_V": 0.0,          # Vertical amplitude (j_hat component)
        "phi_H": 0.0,        # Horizontal phase (radians)
        "phi_V": 0.0,        # Vertical phase (radians)
    },
    {
        "panel": "(b)",
        "A_H": 1.0,          # Horizontal amplitude (i_hat component)
        "A_V": 1.0,          # Vertical amplitude (j_hat component)
        "phi_H": 0.0,        # Horizontal phase (radians)
        "phi_V": np.pi / 4,  # Vertical phase (radians)
    },
]

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)

for ax, case in zip(axes, cases):
    A_H = case["A_H"]
    A_V = case["A_V"]
    phi_H = case["phi_H"]
    phi_V = case["phi_V"]

    # E(0, t) = Re[ (A_H * exp(i*phi_H) i_hat + A_V * exp(i*phi_V) j_hat) * exp(-i*wt) ]
    E_H = np.real(A_H * np.exp(1j * phi_H) * propagator)
    E_V = np.real(A_V * np.exp(1j * phi_V) * propagator)

    ax.plot(wt, E_H, color="#00529B", lw=2.2, label=r"$E_H(0, t)$ ($\hat{i}$)")
    ax.plot(wt, E_V, color="#D9534F", lw=2.2, linestyle="--", label=r"$E_V(0, t)$ ($\hat{j}$)")

    ax.axhline(0, color="black", lw=0.8, alpha=0.5)
    ax.set_xlim(0, 2 * np.pi)
    max_amp = max(A_H, A_V) * 1.25
    ax.set_ylim(-max_amp, max_amp)

    ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
    ax.set_xticklabels([r"$0$", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])

    ax.set_xlabel(r"$\omega t$ (rad)")
    ax.set_title(
        rf"{case['panel']} $A_H = {A_H}$, $A_V = {A_V}$, $\varphi_H = {phi_H/np.pi:.2g}\pi$, $\varphi_V = {phi_V/np.pi:.2g}\pi$"
    )
    ax.grid(True, linestyle=":", alpha=0.6)

axes[0].set_ylabel(r"Electric Field Amplitude (V/m)")
axes[-1].legend(frameon=True, facecolor="white", edgecolor="#CCCCCC", loc="upper right")

plt.tight_layout()
plt.savefig("figure1_2_subplots.pdf", format="pdf", bbox_inches="tight")
plt.show()