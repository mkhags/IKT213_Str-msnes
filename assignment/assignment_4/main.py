import cv2
import numpy as np
from PIL import Image
import sys
import os

def harris_corner_detection(reference_image):
    if len(reference_image.shape) == 3:
        gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = reference_image.copy()
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)
    dst = cv2.dilate(dst, None)
    img_with_corners = reference_image.copy()
    img_with_corners[dst > 0.01 * dst.max()] = [0, 0, 255]
    return img_with_corners

def align_images_sift(image_to_align, reference_image, max_features=10, good_match_percent=0.7, **kwargs):
    """SIFT + FLANN. 'good_match_percent' here is the Lowe ratio threshold (e.g. 0.7)."""
    # alias for the brief's typo just in case
    good_match_percent = kwargs.get("good_match_precent", good_match_percent)

    if not hasattr(cv2, "SIFT_create"):
        raise RuntimeError("cv2.SIFT_create not available. Install opencv-contrib-python or use ORB.")

    im1_gray = cv2.cvtColor(image_to_align, cv2.COLOR_BGR2GRAY)
    im2_gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create(nfeatures=max_features)
    keypoints1, descriptors1 = sift.detectAndCompute(im1_gray, None)
    keypoints2, descriptors2 = sift.detectAndCompute(im2_gray, None)

    if descriptors1 is None or descriptors2 is None:
        raise RuntimeError("Could not compute SIFT descriptors.")

    # FLANN for float descriptors (SIFT)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    knn = flann.knnMatch(descriptors1, descriptors2, k=2)

    good = []
    for m, n in knn:
        if m.distance < good_match_percent * n.distance:  # Lowe ratio test
            good.append(m)

    if len(good) < 4:
        raise RuntimeError(f"Not enough good SIFT matches after ratio test: {len(good)}")

    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if H is None:
        raise RuntimeError("Homography could not be computed with SIFT matches.")

    h, w = reference_image.shape[:2]
    aligned_image = cv2.warpPerspective(image_to_align, H, (w, h))

    inliers = [m for m, inl in zip(good, mask.ravel().tolist()) if inl]
    matches_image = cv2.drawMatches(image_to_align, keypoints1, reference_image, keypoints2,
                                    inliers, None,
                                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return aligned_image, matches_image

def align_images_orb(image_to_align, reference_image, max_features=1500, good_match_percent=0.15, **kwargs):
    """ORB + BF Hamming. 'good_match_percent' here is keep-top-% of matches (e.g. 0.15)."""
    # alias for the brief's typo just in case
    good_match_percent = kwargs.get("good_match_precent", good_match_percent)

    im1_gray = cv2.cvtColor(image_to_align, cv2.COLOR_BGR2GRAY)
    im2_gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=max_features)
    keypoints1, descriptors1 = orb.detectAndCompute(im1_gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2_gray, None)

    if descriptors1 is None or descriptors2 is None:
        raise RuntimeError("Could not compute ORB descriptors.")

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)

    # fixed bug: use sorted(...) not in-place .sort()
    matches = sorted(matches, key=lambda x: x.distance, reverse=False)

    num_good = max(4, int(len(matches) * good_match_percent))
    matches = matches[:num_good]

    if len(matches) < 4:
        raise RuntimeError(f"Not enough ORB matches to compute homography: {len(matches)}")

    points1 = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    points2 = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(points1, points2, cv2.RANSAC, 5.0)
    if H is None:
        raise RuntimeError("Homography could not be computed with ORB matches.")

    h, w = reference_image.shape[:2]
    aligned_image = cv2.warpPerspective(image_to_align, H, (w, h))

    inliers = [m for m, inl in zip(matches, mask.ravel().tolist()) if inl]
    matches_image = cv2.drawMatches(image_to_align, keypoints1, reference_image, keypoints2,
                                    inliers, None,
                                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return aligned_image, matches_image

def save_pdf(p1, p2, p3, pdf_path="assignment4_output.pdf"):
    imgs = []
    for p in [p1, p2, p3]:
        if not os.path.isfile(p):
            raise FileNotFoundError(p)
        im = Image.open(p)
        if im.mode != "RGB":
            im = im.convert("RGB")
        imgs.append(im)
    cover, *rest = imgs
    cover.save(pdf_path, save_all=True, append_images=rest)
    return pdf_path

if __name__ == "__main__":
    reference_img = cv2.imread('reference_img.png')
    align_this_img = cv2.imread('align_this.jpg')

    if reference_img is None:
        print("Error: Could not load reference_img.png"); sys.exit(1)
    if align_this_img is None:
        print("Error: Could not load align_this.jpg"); sys.exit(1)

    # 1) Harris (page 1)
    harris_result = harris_corner_detection(reference_img)
    cv2.imwrite('harris.png', harris_result)

    # 2) Choose ONE method and use the assignment’s exact parameters:
    method = 'ORB'   # change to 'SIFT' if you pick that approach

    if method == 'SIFT':
        aligned, matches_img = align_images_sift(
            align_this_img, reference_img,
            max_features=10,           # as specified
            good_match_percent=0.7     # Lowe ratio threshold
        )
    else:
        aligned, matches_img = align_images_orb(
            align_this_img, reference_img,
            max_features=1500,         # as specified
            good_match_percent=0.15    # keep top 15% matches
        )

    cv2.imwrite('aligned.png', aligned)
    cv2.imwrite('matches.png', matches_img)

    # 3) Make the 3-page PDF in the required order
    pdf = save_pdf('harris.png', 'aligned.png', 'matches.png')
    print(f"[OK] Saved PDF -> {pdf}")
