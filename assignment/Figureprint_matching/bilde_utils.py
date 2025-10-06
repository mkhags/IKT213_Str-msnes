import cv2
import numpy as np

def last_inn_bilde(sti: str) -> np.ndarray:
    return cv2.imread(str(sti), cv2.IMREAD_GRAYSCALE)

def binær_konvertering(bilde: np.ndarray) -> np.ndarray:
    _, resultat = cv2.threshold(
        bilde, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    return resultat