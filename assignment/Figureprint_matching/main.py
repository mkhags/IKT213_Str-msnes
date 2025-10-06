from pathlib import Path
from bilde_utils import last_inn_bilde, binær_konvertering
from orb_modul import orb_pipeline
from sift_modul import sift_pipeline
from analyse import sammenlign_resultater
from fil_håndtering import skriv_rapport, lagre_plot


def kjør_matching(bilde1_sti, bilde2_sti):
    """Hovedfunksjon"""
    print(f"\n{'=' * 60}")
    print(f"Matcher: {Path(bilde1_sti).name} vs {Path(bilde2_sti).name}")
    print(f"{'=' * 60}")

    # Last og forbered
    bilde1 = binær_konvertering(last_inn_bilde(bilde1_sti))
    bilde2 = binær_konvertering(last_inn_bilde(bilde2_sti))

    # Kjør begge metoder
    orb_data = orb_pipeline(bilde1, bilde2)
    sift_data = sift_pipeline(bilde1, bilde2)

    # Analyser
    resultat = sammenlign_resultater(orb_data, sift_data)

    # Lagre
    skriv_rapport(resultat, bilde1_sti, bilde2_sti)
    lagre_plot(resultat, bilde1_sti, bilde2_sti)


if __name__ == "__main__":
    kjør_matching("pictures/UiA front1.png", "pictures/UiA front3.jpg")