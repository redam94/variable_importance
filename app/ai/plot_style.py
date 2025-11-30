"""
Professional Matplotlib Styling Configuration

Provides a modern, clean aesthetic for data visualizations.
Auto-injected before code execution for consistent chart appearance.
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
from cycler import cycler


# Color palette - purple gradient theme with complementary colors
COLORS = {
    "primary": "#667eea",      # Purple-blue
    "secondary": "#764ba2",    # Deep purple  
    "accent": "#f093fb",       # Light magenta
    "success": "#48bb78",      # Green
    "warning": "#ed8936",      # Orange
    "error": "#fc8181",        # Red
    "neutral": "#718096",      # Gray
    "dark": "#2d3748",         # Dark gray
    "light": "#edf2f7",        # Light gray
}

# Extended palette for multi-series charts
PALETTE = [
    "#667eea",  # Primary purple-blue
    "#48bb78",  # Green
    "#ed8936",  # Orange
    "#f56565",  # Red
    "#38b2ac",  # Teal
    "#9f7aea",  # Violet
    "#ed64a6",  # Pink
    "#4299e1",  # Blue
    "#ecc94b",  # Yellow
    "#718096",  # Gray
]


def apply_style():
    """Apply professional styling to all matplotlib figures."""
    
    # Reset to default first
    plt.style.use('default')
    
    # Core style settings
    style_config = {
        # Figure
        "figure.figsize": (10, 6),
        "figure.dpi": 150,
        "figure.facecolor": "white",
        "figure.edgecolor": "white",
        "figure.autolayout": True,
        
        # Axes
        "axes.facecolor": "white",
        "axes.edgecolor": COLORS["neutral"],
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.labelsize": 11,
        "axes.labelcolor": COLORS["dark"],
        "axes.labelweight": "medium",
        "axes.titlesize": 13,
        "axes.titleweight": "semibold",
        "axes.titlecolor": COLORS["dark"],
        "axes.titlepad": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.prop_cycle": cycler("color", PALETTE),
        
        # Grid
        "grid.color": "#e2e8f0",
        "grid.linewidth": 0.5,
        "grid.alpha": 0.7,
        
        # Lines
        "lines.linewidth": 2,
        "lines.markersize": 6,
        
        # Patches (bars, etc.)
        "patch.linewidth": 0,
        "patch.edgecolor": "white",
        
        # Legend
        "legend.frameon": True,
        "legend.framealpha": 0.95,
        "legend.facecolor": "white",
        "legend.edgecolor": "#e2e8f0",
        "legend.fontsize": 10,
        "legend.title_fontsize": 11,
        "legend.borderpad": 0.6,
        "legend.labelspacing": 0.5,
        
        # Ticks
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "xtick.color": COLORS["neutral"],
        "ytick.color": COLORS["neutral"],
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        
        # Font
        "font.family": "sans-serif",
        "font.sans-serif": ["Inter", "Segoe UI", "Helvetica", "Arial", "sans-serif"],
        "font.size": 10,
        
        # Savefig
        "savefig.dpi": 300,
        "savefig.facecolor": "white",
        "savefig.edgecolor": "white",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.15,
    }
    
    for key, value in style_config.items():
        try:
            mpl.rcParams[key] = value
        except KeyError:
            pass  # Skip unsupported params


def style_figure(fig=None, ax=None, title=None, xlabel=None, ylabel=None):
    """
    Apply additional styling to a specific figure/axes.
    
    Args:
        fig: Figure object (optional)
        ax: Axes object (optional)  
        title: Chart title
        xlabel: X-axis label
        ylabel: Y-axis label
    """
    if fig is None:
        fig = plt.gcf()
    if ax is None:
        ax = plt.gca()
    
    # Set labels if provided
    if title:
        ax.set_title(title, fontweight="semibold", pad=12)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    
    # Subtle shadow effect via light border
    for spine in ["bottom", "left"]:
        ax.spines[spine].set_color(COLORS["neutral"])
        ax.spines[spine].set_linewidth(0.8)
    
    fig.tight_layout()
    return fig, ax


def save_plot(filename, fig=None, **kwargs):
    """
    Save plot with professional settings.
    
    Args:
        filename: Output filename
        fig: Figure to save (defaults to current)
        **kwargs: Additional savefig arguments
    """
    if fig is None:
        fig = plt.gcf()
    
    save_kwargs = {
        "dpi": 300,
        "facecolor": "white",
        "edgecolor": "none",
        "bbox_inches": "tight",
        "pad_inches": 0.15,
    }
    save_kwargs.update(kwargs)
    
    fig.savefig(filename, **save_kwargs)
    plt.close(fig)


# Convenience functions for common chart types

def bar_colors(n=None):
    """Get colors for bar charts."""
    if n is None:
        return PALETTE
    return PALETTE[:n]


def get_color(name):
    """Get a named color from the palette."""
    return COLORS.get(name, COLORS["primary"])


# Auto-apply on import
apply_style()