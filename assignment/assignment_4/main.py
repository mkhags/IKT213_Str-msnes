import cv2
import numpy as np


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


def align_images_sift(image_to_align, reference_image, max_features=10, good_match_percent=0.7):
    im1_gray = cv2.cvtColor(image_to_align, cv2.COLOR_BGR2GRAY)
    im2_gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create(max_features)
    keypoints1, descriptors1 = sift.detectAndCompute(im1_gray, None)
    keypoints2, descriptors2 = sift.detectAndCompute(im2_gray, None)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    good_matches = sorted(good_matches, key=lambda x: x.distance, reverse=False)

    num_good_matches = int(len(good_matches) * good_match_percent)
    good_matches = good_matches[:num_good_matches]

    matches_image = cv2.drawMatches(image_to_align, keypoints1, reference_image,
                                    keypoints2, good_matches, None,
                                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    points1 = np.zeros((len(good_matches), 2), dtype=np.float32)
    points2 = np.zeros((len(good_matches), 2), dtype=np.float32)

    for i, match in enumerate(good_matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    height, width, channels = reference_image.shape
    aligned_image = cv2.warpPerspective(image_to_align, h, (width, height))

    return aligned_image, matches_image


def align_images_orb(image_to_align, reference_image, max_features=1500, good_match_percent=0.15):
    im1_gray = cv2.cvtColor(image_to_align, cv2.COLOR_BGR2GRAY)
    im2_gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(max_features)
    keypoints1, descriptors1 = orb.detectAndCompute(im1_gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2_gray, None)

    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)

    matches = sorted(matches, key=lambda x: x.distance, reverse=False)

    num_good_matches = int(len(matches) * good_match_percent)
    matches = matches[:num_good_matches]

    matches_image = cv2.drawMatches(image_to_align, keypoints1, reference_image,
                                    keypoints2, matches, None,
                                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC, 5.0)

    height, width, channels = reference_image.shape
    aligned_image = cv2.warpPerspective(image_to_align, h, (width, height))

    return aligned_image, matches_image


if __name__ == "__main__":
    reference_img = cv2.imread('reference_img.png')
    align_this_img = cv2.imread('align_this.jpg')

    if reference_img is None:
        print("Error: Could not load reference_img.png")
        exit(1)
    if align_this_img is None:
        print("Error: Could not load align_this.jpg")
        exit(1)

    harris_result = harris_corner_detection(reference_img)
    cv2.imwrite('harris.png', harris_result)

    method = 'ORB'

    if method == 'SIFT':
        aligned, matches_img = align_images_sift(align_this_img, reference_img,
                                                 max_features=5000,
                                                 good_match_percent=0.15)
    else:
        aligned, matches_img = align_images_orb(align_this_img, reference_img,
                                                max_features=5000,
                                                good_match_percent=0.20)

    cv2.imwrite('aligned.png', aligned)
    cv2.imwrite('matches.png', matches_img)