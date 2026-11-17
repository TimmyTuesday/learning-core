"""matplotlib-Stil „Scientific Noir“ – passend zum Design-System des Learning Core."""
import matplotlib as mpl

BG, PANEL, HAIRLINE = "#05070A", "#0B0F14", "#1E2A33"
INK, INK_DIM = "#E8ECEF", "#8A97A3"
ACCENT, ACCENT_SOFT, COOL, CREAM = "#FF4A1C", "#FF8A5C", "#4FB3C8", "#E9DDB5"


def use_noir() -> None:
    """Aktiviert den Stil global für alle folgenden Plots."""
    mpl.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
        "axes.edgecolor": HAIRLINE, "axes.linewidth": 0.5,
        "axes.labelcolor": INK_DIM, "axes.titlecolor": INK,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": HAIRLINE, "grid.linewidth": 0.5,
        "xtick.color": INK_DIM, "ytick.color": INK_DIM,
        "text.color": INK, "font.family": "monospace", "font.size": 10,
        "axes.prop_cycle": mpl.cycler(color=[CREAM, ACCENT, COOL, INK_DIM]),
        "lines.linewidth": 1.0,
    })
