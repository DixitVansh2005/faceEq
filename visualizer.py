"""
visualizer.py
-------------
Generates all visual outputs for the FaceIQ dashboard.

Functions:
  - emotion_bar_chart()  : horizontal bar chart of emotion scores
  - race_bar_chart()     : horizontal bar chart of ethnicity scores
  - annotate_face()      : draws bounding box + label on the original image
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io


# Colour palette — consistent across all charts
PALETTE = {
    'primary':    '#00d4ff',
    'secondary':  '#7b2fff',
    'accent':     '#ff2d7a',
    'success':    '#00ff88',
    'warning':    '#ffaa00',
    'muted1':     '#ff6b35',
    'muted2':     '#a0a0c0',
    'bg':         '#11111a',
    'bar_muted':  '#2a2a4a',
    'text':       '#8888a0',
    'subtext':    '#4a4a6a',
}

CHART_COLORS = [
    PALETTE['primary'], PALETTE['secondary'], PALETTE['accent'],
    PALETTE['success'], PALETTE['warning'], PALETTE['muted1'], PALETTE['muted2']
]


def _fig_to_pil(fig) -> Image.Image:
    """Convert a matplotlib figure to a PIL Image."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=130, bbox_inches='tight',
                facecolor=PALETTE['bg'])
    buf.seek(0)
    plt.close(fig)
    return Image.open(buf)


def emotion_bar_chart(emotions: dict) -> Image.Image:
    """
    Horizontal bar chart showing all 7 emotion confidence scores.

    Parameters
    ----------
    emotions : dict  — e.g. {'happy': 92.3, 'sad': 3.1, ...}

    Returns
    -------
    PIL Image
    """
    sorted_em = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
    labels = [e[0].capitalize() for e in sorted_em]
    values = [e[1] for e in sorted_em]

    fig, ax = plt.subplots(figsize=(5, 3))
    fig.patch.set_facecolor(PALETTE['bg'])
    ax.set_facecolor(PALETTE['bg'])

    bars = ax.barh(labels, values,
                   color=CHART_COLORS[:len(labels)],
                   height=0.52, alpha=0.88)

    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.8,
                bar.get_y() + bar.get_height() / 2,
                f'{val:.1f}%',
                va='center', ha='left',
                color=PALETTE['subtext'], fontsize=8,
                fontfamily='monospace')

    ax.set_xlim(0, 115)
    ax.xaxis.set_visible(False)
    ax.tick_params(colors=PALETTE['subtext'], labelsize=9)
    plt.yticks(color=PALETTE['text'], fontsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout(pad=0.4)
    return _fig_to_pil(fig)


def race_bar_chart(race_scores: dict) -> Image.Image:
    """
    Horizontal bar chart for ethnicity confidence scores.

    Parameters
    ----------
    race_scores : dict  — e.g. {'asian': 45.2, 'white': 30.1, ...}

    Returns
    -------
    PIL Image
    """
    sorted_r = sorted(race_scores.items(), key=lambda x: x[1], reverse=True)
    labels = [r[0].capitalize() for r in sorted_r]
    values = [r[1] for r in sorted_r]

    # Dominant bar highlighted, rest muted
    colors = [PALETTE['primary']] + [PALETTE['bar_muted']] * (len(labels) - 1)

    fig, ax = plt.subplots(figsize=(5, 2.8))
    fig.patch.set_facecolor(PALETTE['bg'])
    ax.set_facecolor(PALETTE['bg'])

    ax.barh(labels, values, color=colors, height=0.52, alpha=0.88)

    for i, (val, label) in enumerate(zip(values, labels)):
        ax.text(val + 0.8, i,
                f'{val:.1f}%',
                va='center', ha='left',
                color=PALETTE['subtext'], fontsize=8,
                fontfamily='monospace')

    ax.set_xlim(0, 115)
    ax.xaxis.set_visible(False)
    ax.tick_params(colors=PALETTE['subtext'], labelsize=9)
    plt.yticks(color=PALETTE['text'], fontsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout(pad=0.4)
    return _fig_to_pil(fig)


def annotate_face(img_array: np.ndarray, region: dict, label: str) -> Image.Image:
    """
    Draw a bounding box and label on the image.

    Parameters
    ----------
    img_array : np.ndarray — original RGB image
    region    : dict       — DeepFace region dict with x, y, w, h keys
    label     : str        — text to show above the box

    Returns
    -------
    PIL Image with annotation
    """
    image = Image.fromarray(img_array.astype(np.uint8)).copy()
    draw = ImageDraw.Draw(image)

    x = region.get('x', 0)
    y = region.get('y', 0)
    w = region.get('w', 0)
    h = region.get('h', 0)

    if w > 0 and h > 0:
        # Box
        box_color = (0, 212, 255)   # cyan
        draw.rectangle([x, y, x + w, y + h],
                       outline=box_color, width=3)

        # Label background
        label_y = max(y - 28, 0)
        draw.rectangle([x, label_y, x + w, label_y + 24],
                       fill=(17, 17, 26))

        # Label text
        draw.text((x + 6, label_y + 4), label,
                  fill=box_color)

    return image