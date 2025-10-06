import cv2
import time


def orb_pipeline(bilde1, bilde2, antall_features=1000, ratio=0.7):
    start = time.time()

    # Ekstraksjon
    orb = cv2.ORB_create(nfeatures=antall_features)
    kp1, des1 = orb.detectAndCompute(bilde1, None)
    kp2, des2 = orb.detectAndCompute(bilde2, None)

    # Matching
    if des1 is None or des2 is None:
        return None, [], 0, 0, time.time() - start

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    råmatches = bf.knnMatch(des1, des2, k=2)

    # Filtrer
    gode = []
    for par in råmatches:
        if len(par) == 2:
            m, n = par
            if m.distance < ratio * n.distance:
                gode.append(m)

    # Visualisering
    vis_bilde = cv2.drawMatches(
        bilde1, kp1, bilde2, kp2, gode, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    return vis_bilde, gode, len(kp1), len(kp2), time.time() - start