import cv2
import numpy as np

def harris_corner_detection(reference_image):
    if len(reference_image.shape) == 3:
        gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = reference_image.copy()
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)
    dst = cv2.dilate(dst, None)
    out = reference_image.copy()
    out[dst > 0.01 * dst.max()] = [0, 0, 255]
    return out

def align_images_orb(image_to_align, reference_image, max_features=1500, good_match_precent=0.15):
    g1 = cv2.cvtColor(image_to_align, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=max_features)
    kp1, des1 = orb.detectAndCompute(g1, None)
    kp2, des2 = orb.detectAndCompute(g2, None)
    if des1 is None or des2 is None:
        raise SystemExit(1)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    k = max(8, int(len(matches) * good_match_precent))
    matches = matches[:k]
    if len(matches) < 8:
        raise SystemExit(1)
    src = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    if H is None or mask is None or mask.sum() < 6:
        raise SystemExit(1)
    h, w = reference_image.shape[:2]
    aligned = cv2.warpPerspective(image_to_align, H, (w, h))
    inliers = [m for m, inl in zip(matches, mask.ravel().tolist()) if inl]
    matches_img = cv2.drawMatches(image_to_align, kp1, reference_image, kp2, inliers, None,
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return aligned, matches_img

if __name__ == "__main__":
    reference_img = cv2.imread("reference_img.png")
    image_to_align = cv2.imread("align_this.jpg")
    if reference_img is None or image_to_align is None:
        raise SystemExit(1)
    harris = harris_corner_detection(reference_img)
    cv2.imwrite("harris.png", harris)
    aligned, matches_img = align_images_orb(image_to_align, reference_img, max_features=1500, good_match_precent=0.15)
    cv2.imwrite("aligned.png", aligned)
    cv2.imwrite("matches.png", matches_img)