import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime


def skriv_rapport(data, bilde1_navn, bilde2_navn):
    Path("results").mkdir(exist_ok=True)

    tid = datetime.now().strftime("%Y%m%d_%H%M%S")
    fil = Path("results") / f"rapport_{tid}.txt"

    with open(fil, 'w', encoding='utf-8') as f:
        f.write("FINGERAVTRYKK MATCHING RAPPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Dato: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Bilder: {bilde1_navn} vs {bilde2_navn}\n\n")

        f.write("ORB+BF METODE:\n")
        f.write(f"  Matches: {data['orb']['matches']}\n")
        f.write(f"  Keypoints: {data['orb']['kp1']}, {data['orb']['kp2']}\n")
        f.write(f"  Tid: {data['orb']['tid']:.4f}s\n\n")

        f.write("SIFT+FLANN METODE:\n")
        f.write(f"  Matches: {data['sift']['matches']}\n")
        f.write(f"  Keypoints: {data['sift']['kp1']}, {data['sift']['kp2']}\n")
        f.write(f"  Tid: {data['sift']['tid']:.4f}s\n")

    print(f"Rapport lagret: {fil}")


def lagre_plot(data, bilde1_navn, bilde2_navn):
    """Lagre visualisering"""
    fig, ax = plt.subplots(2, 1, figsize=(16, 12))

    if data['orb']['bilde'] is not None:
        ax[0].imshow(cv2.cvtColor(data['orb']['bilde'], cv2.COLOR_BGR2RGB))
        ax[0].set_title(f"ORB+BF: {data['orb']['matches']} matches", fontsize=14)
        ax[0].axis('off')

    if data['sift']['bilde'] is not None:
        ax[1].imshow(cv2.cvtColor(data['sift']['bilde'], cv2.COLOR_BGR2RGB))
        ax[1].set_title(f"SIFT+FLANN: {data['sift']['matches']} matches", fontsize=14)
        ax[1].axis('off')

    plt.tight_layout()

    Path("results").mkdir(exist_ok=True)
    fil = Path("results") / f"visualisering_{Path(bilde1_navn).stem}_{Path(bilde2_navn).stem}.png"
    plt.savefig(fil, dpi=150, bbox_inches='tight')
    print(f"Plot lagret: {fil}")
    plt.show()