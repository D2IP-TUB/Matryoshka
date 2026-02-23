import matplotlib as mpl

def apply_style():
    mpl.rcParams.update({

        # --- Figure size ---
        "figure.figsize": (3.3, 2.2),
        "figure.dpi": 300,

        # --- Fonts ---
        "font.size": 12,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],

        # --- Axes ---
        "axes.titlesize": 10,
        "axes.labelsize": 10,
        "axes.linewidth": 0.8,
        "axes.grid": False,

        # --- Lines ---
        "lines.linewidth": 1.5,
        "lines.markersize": 5,

        # --- Colors (grayscale-compatible palette) ---
        "axes.prop_cycle": mpl.cycler("color", [
            "black",
            "dimgray",
            "lightgray",
            "#4c72b0",
            "#dd8452",
        ]),

        # --- Legend ---
        "legend.fontsize": 9,
        "legend.frameon": False,
        "legend.handlelength": 2,

        # --- Ticks ---
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,

        # --- Savefig ---
        "savefig.bbox": "tight",
        "savefig.dpi": 300,
    })