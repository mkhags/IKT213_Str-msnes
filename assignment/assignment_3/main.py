import cv2
import numpy as np
import os


def sobel_edge_detection(bilde_inn):
    graa = cv2.cvtColor(bilde_inn, cv2.COLOR_BGR2GRAY)

    utglatt = cv2.GaussianBlur(graa, (3, 3), sigmaX=0)
    kant_resultat = cv2.Sobel(utglatt, cv2.CV_64F, dx=1, dy=1, ksize=1)

    kant_resultat = np.absolute(kant_resultat)
    ferdig_kanter = np.uint8(kant_resultat)

    cv2.imwrite('output/sobel_edge_detection.png', ferdig_kanter)
    return ferdig_kanter


def canny_edge_detection(bilde_inn, lav_grense, hoy_grense):
    graa_bilde = cv2.cvtColor(bilde_inn, cv2.COLOR_BGR2GRAY)

    blur_bilde = cv2.GaussianBlur(graa_bilde, (3, 3), sigmaX=0)

    kanter = cv2.Canny(blur_bilde, lav_grense, hoy_grense)
    cv2.imwrite('output/canny_edge_detection.png', kanter)

    return kanter


def template_match(kilde_bilde, monster_bilde):
    kilde_graa = cv2.cvtColor(kilde_bilde, cv2.COLOR_BGR2GRAY)
    monster_graa = cv2.cvtColor(monster_bilde, cv2.COLOR_BGR2GRAY)

    monster_hoyde, monster_bredde = monster_graa.shape
    match_resultat = cv2.matchTemplate(kilde_graa, monster_graa, cv2.TM_CCOEFF_NORMED)

    terskel = 0.9
    funn_posisjoner = np.where(match_resultat >= terskel)

    ut_bilde = kilde_bilde.copy()

    for punkt in zip(*funn_posisjoner[::-1]):
        cv2.rectangle(ut_bilde, punkt, (punkt[0] + monster_bredde, punkt[1] + monster_hoyde), (0, 0, 255), 2)

    cv2.imwrite('output/template_match_result.png', ut_bilde)
    return ut_bilde


def resize(inn_bilde, faktor, retning):
    arbeids_bilde = inn_bilde.copy()

    if retning.lower() == "up":
        for steg in range(faktor):
            arbeids_bilde = cv2.pyrUp(arbeids_bilde)

        sti = f'output/resized_up_scale_{faktor}.png'

    elif retning.lower() == "down":
        for steg in range(faktor):
            arbeids_bilde = cv2.pyrDown(arbeids_bilde)

        sti = f'output/resized_down_scale_{faktor}.png'

    cv2.imwrite(sti, arbeids_bilde)
    return arbeids_bilde


def main():
    os.makedirs('output', exist_ok=True)

    bil_bilde = cv2.imread('lambo.png')

    sobel_edge_detection(bil_bilde)

    canny_edge_detection(bil_bilde, 50, 50)

    former_bilde = cv2.imread('shapes-1.png')
    mal_bilde = cv2.imread('shapes_template.jpg')

    template_match(former_bilde, mal_bilde)

    resize(bil_bilde, 2, "up")
    resize(bil_bilde, 2, "down")


if __name__ == "__main__":
    main()