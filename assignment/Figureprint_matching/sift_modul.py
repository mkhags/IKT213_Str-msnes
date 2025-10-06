import cv2
import time


def sift_pipeline(bilde1, bilde2, antall_features=1000, ratio=0.7):
    start = time.time()

    # Ekstraksjon
    sift = cv2.SIFT_create(nfeatures=antall_features)
    kp1, des1 = sift.detectAndCompute(bilde1, None)
    kp2, des2 = sift.detectAndCompute(bilde2, None)

    # Matching
    if des1 is None or des2 is None:
        return None, [], 0, 0, time.time() - start

    index_p = dict(algorithm=1, trees=5)
    search_p = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_p, search_p)
    råmatches = flann.knnMatch(des1, des2, k=2)

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