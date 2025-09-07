import numpy as np
import cv2
import os

# hjelpe_funksjoner
def last_bilde(filnavn: str) -> np.ndarray:
    return cv2.imread(os.path.join("picture", filnavn))

def lagre_bilde(bilde: np.ndarray, filnavn: str):
    cv2.imwrite(os.path.join("output", filnavn), bilde)


# oppgave_1
# bruker copyMakeBorder fra cv2, den lager kant rundt bildet
# her brukes REFLECT så kanten på bildet kan bli speilvendt av bildet
def padding(bilde: np.ndarray, kant: int) -> np.ndarray:
    return cv2.copyMakeBorder(bilde, kant, kant, kant, kant, cv2.BORDER_REFLECT)

# oppgave_2
# cropping gjøres bare med numpy slicing → bilde[y0:y1, x0:x1]
def crop(bilde: np.ndarray, x0: int, x1: int, y0: int, y1: int) -> np.ndarray:
    return bilde[y0:y1, x0:x1]

# oppgave_3
# resize med cv2.resize, gir (bredde,høyde) og type interpolasjon
def resize(bilde: np.ndarray, bredde: int, hoyde: int) -> np.ndarray:
    return cv2.resize(bilde, (bredde, hoyde), interpolation=cv2.INTER_LINEAR)

# oppgave_4
# kopierer bildet ved hjelp av funksjonen bilde.copy() fra numpy
def kopi(bilde: np.ndarray) -> np.ndarray:
    return bilde.copy()

# oppgave_5
# Endrer fargen på bildet ved hjelp av funksjonen cv2.cvtColor()
def grattone(bilde: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bilde, cv2.COLOR_BGR2GRAY)

# oppgave_6
# Endrer fargen på bildet
def til_hsv(bilde: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bilde, cv2.COLOR_BGR2HSV)

# oppgave_7
# Endrer farge på bildet
def hue_shift(bilde: np.ndarray, forskyvning: int) -> np.ndarray:
    hsv_img = cv2.cvtColor(bilde, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv_img)
    H = ((H.astype(np.int16) + forskyvning) % 180).astype(np.uint8)
    sammen = cv2.merge([H, S, V])
    return cv2.cvtColor(sammen, cv2.COLOR_HSV2BGR)

# oppgave_8
# Smoothing her er GaussianBlur, gir mer naturlig blur enn vanlig box
def glatting(bilde: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(bilde, (15, 15), sigmaX=0)

# oppgave 9
# cv2.rotate håndterer både 90 og 180 grader ferdig
def rotering(bilde: np.ndarray, vinkel: int) -> np.ndarray:
    if vinkel == 90:
        return cv2.rotate(bilde, cv2.ROTATE_90_CLOCKWISE)
    elif vinkel == 180:
        return cv2.rotate(bilde, cv2.ROTATE_180)
    return bilde

if __name__ == "__main__":
    inndata = "lena-2.png"
    bilde = last_bilde(inndata)

    lagre_bilde(padding(bilde, 100), "oppg1_pad.png")
    lagre_bilde(crop(bilde, 80, bilde.shape[1]-130, 80, bilde.shape[0]-130), "oppg2_crop.png")
    lagre_bilde(resize(bilde, 200, 200), "oppg3_resize.png")
    lagre_bilde(kopi(bilde), "oppg4_copy.png")
    lagre_bilde(grattone(bilde), "oppg5_gray.png")
    lagre_bilde(til_hsv(bilde), "oppg6_hsv.png")
    lagre_bilde(hue_shift(bilde, 50), "oppg7_hue.png")
    lagre_bilde(glatting(bilde), "oppg8_blur.png")
    lagre_bilde(rotering(bilde, 90), "oppg9_rot90.png")
    lagre_bilde(rotering(bilde, 180), "oppg9_rot180.png")