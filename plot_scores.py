import pandas as pd
import matplotlib.pyplot as plt

# Results from score_ft_a100.txt
data = {
    "noise": [
        "airplane", "drone", "engine", "explosion", "field",
        "fireworks", "gunshot", "gunshot_real", "helicopter",
        "mad_fighter", "mad_gunshot", "mad_helicopter",
        "mad_shelling", "mad_vehicle", "siren", "traffic",
        "vehicle", "wind"
    ],

    "PESQ_in": [
        1.31, 1.22, 1.28, 1.62, 1.61, 1.45, 1.35, 1.61,
        1.35, 1.22, 1.27, 1.25, 1.27, 1.34, 1.40, 1.44,
        1.62, 1.49
    ],

    "PESQ_out": [
        2.96, 2.95, 2.93, 3.38, 3.27, 3.25, 2.99, 3.53,
        2.89, 2.87, 2.79, 2.79, 2.82, 3.00, 3.41, 3.31,
        3.41, 3.37
    ],

    "STOI_in": [
        0.848, 0.847, 0.840, 0.884, 0.903, 0.867, 0.827, 0.874,
        0.834, 0.811, 0.811, 0.812, 0.808, 0.853, 0.855, 0.894,
        0.889, 0.903
    ],

    "STOI_out": [
        0.938, 0.935, 0.924, 0.958, 0.957, 0.950, 0.924, 0.965,
        0.932, 0.922, 0.924, 0.913, 0.919, 0.942, 0.952, 0.950,
        0.955, 0.965
    ],

    "SNR_in": [
        6.7, 8.2, 7.3, 8.1, 7.4, 8.0, 7.3, 7.4, 7.7,
        7.1, 7.3, 7.6, 7.1, 8.3, 7.0, 8.5, 7.4, 7.8
    ],

    "SNR_out": [
        17.6, 18.9, 18.2, 20.4, 19.7, 20.1, 19.4, 20.7, 18.2,
        17.9, 18.6, 17.8, 18.0, 18.4, 20.7, 19.5, 18.6, 18.9
    ]
}

df = pd.DataFrame(data)

x = range(len(df))

fig, axes = plt.subplots(3, 1, figsize=(16, 14))

# ---------------- PESQ ----------------
axes[0].plot(x, df["PESQ_in"], "o-", label="Input", linewidth=2)
axes[0].plot(x, df["PESQ_out"], "o-", label="Enhanced", linewidth=2)

axes[0].set_ylabel("PESQ")
axes[0].set_title("PESQ — Input vs Enhanced")
axes[0].legend()
axes[0].grid(alpha=0.3)

# ---------------- STOI ----------------
axes[1].plot(x, df["STOI_in"], "o-", label="Input", linewidth=2)
axes[1].plot(x, df["STOI_out"], "o-", label="Enhanced", linewidth=2)

axes[1].set_ylabel("STOI")
axes[1].set_title("STOI — Input vs Enhanced")
axes[1].legend()
axes[1].grid(alpha=0.3)

# ---------------- SNR ----------------
axes[2].plot(x, df["SNR_in"], "o-", label="Input", linewidth=2)
axes[2].plot(x, df["SNR_out"], "o-", label="Enhanced", linewidth=2)

axes[2].set_ylabel("SNR (dB)")
axes[2].set_title("SNR — Input vs Enhanced")
axes[2].set_xticks(list(x))
axes[2].set_xticklabels(df["noise"], rotation=60, ha="right")
axes[2].legend()
axes[2].grid(alpha=0.3)

plt.tight_layout()

# Save high-resolution figure
plt.savefig(
    "score_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

print("Saved: score_comparison.png")

plt.show()
