#!/usr/bin/evn python

"""
RBE/CS Fall 2022: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 1: MyAutoPano: Phase 2 Starter Code


Author(s):
Lening Li (lli4@wpi.edu)
Teaching Assistant in Robotics Engineering,
Worcester Polytechnic Institute
"""

# Code starts here:

import numpy as np
import cv2
from numba.core.cgutils import raw_memcpy
from skimage.feature import peak_local_max
import os
import matplotlib.pyplot as plt


def show_helper(inputs, tiled=True, title=None):
    if tiled:
        fig, axes = plt.subplots(1, len(inputs))
        for i, img in enumerate(inputs):
            axes[i].imshow(img)
            axes[i].axis('off')
            if title:
                axes[i].set_title(title)
        plt.show()
    else:
        for img in inputs:
            plt.imshow(img)
            plt.axis('off')
            if title:
                plt.title(title)
            plt.show()


# Add any python libraries here
def load_images(path: str, set: str, show: bool = False):
    images = []
    base_path = os.path.join(path, set)
    for img in os.listdir(base_path):
        image = cv2.imread(os.path.join(base_path, img), cv2.COLOR_BGR2RGB)
        image = np.uint8(image)
        images.append(image)
    if show:
        show_helper(images)

    return images


def detect_corners(images, method: str, max_corners=100, show: bool = False):
    """
    Corner Detection
    :param images: list of images
    :param method: corner detection method: either 'Harris' or 'Shi-Tomasi'
    :param show: show the corner detection output
    """
    outputs = []
    # convert to grayscale
    grayscales = [cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) for img in images]
    if method == 'Harris':
        for img in grayscales:
            corners = cv2.cornerHarris(img, 4, 3, 0.04)
            outputs.append(corners)
    elif method == 'Shi-Tomasi':
        for img in grayscales:
            corners = cv2.goodFeaturesToTrack(img, maxCorners=max_corners, qualityLevel=0.01, minDistance=20)
            corners = np.int32(corners)
            outputs.append(corners)
    else:
        raise ValueError("Invalid corner detection method")
    if show:
        corners = []
        for img, dst in zip(images, outputs):
            # Make a copy of the image as to not modify the original
            img = img.copy()
            if 'Shi-Tomasi' == method:
                # dst = cv2.dilate(dst, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
                # dst = dst.astype(np.int32)
                for i in dst:
                    x, y = i.ravel()
                    cv2.circle(img, (x, y), 3, 255, 3)
                corners.append(img)
            else:
                img[dst > 0.01 * dst.max()] = [255, 0, 0]
                corners.append(img)
        show_helper(corners, False)
    return outputs


def run_ANMS(images, corner_scores, num_best_corners, method: str, show: bool = False):
    """
    Run Adaptive Non-Maximal Suppression
    :param corner_scores: list of corner scores
    :param num_best_corners: number of best corners to keep
    :param method: method to run ANMS: either 'Harris' or 'Shi-Tomasi'
    :param show: show the ANMS output
    """
    # Preprocess according to the method
    corner_indices = []
    if len(corner_scores) != len(images):
        raise ValueError("Number of corner scores and images do not match")
    # Making them into a list of lists to make the for loops not kill themselves
    if len(corner_scores) == 1:
        corner_scores = [corner_scores]
        images = [images]
    if method == 'Harris':
        for score in corner_scores:
            # score = peak_local_max(score, min_distance=5)
            # corner_indices.append(score)
            corner_indices.append(np.argwhere(score > 0.05 * score.max()))
    elif method == 'Shi-Tomasi':
        print("Shi-Tomasi already does ANMS")
        return corner_scores
        # for score in corner_scores:
        #     corner_indices.append([i.ravel().astype(np.uint32) for i in score])
    else:
        raise ValueError("Invalid corner detection method")

    # Run ANMS
    anms_outputs = []
    ## This is the original code that uses nested
    # for c_score, indices in zip(corner_scores, corner_indices):
    #     ranking = np.inf * np.ones(len(indices))
    #     # Reverse the order of the indices since i is for rows (y) and j is for columns (x)
    #     for i, (y_i, x_i) in enumerate(indices):
    #         for j, (y_j, x_j) in enumerate(indices):
    #             if i == j:
    #                 continue
    #             if c_score[y_j, x_j] > c_score[y_i, x_i]:
    #                 ranking[i] = min(ranking[i], np.power(x_j - x_i, 2) + np.power(y_j - y_i, 2))

    ## This is the optimized version of the above code
    ## Performed by ChatGPT by using the propmt:
    ## Convert this into the efficient numpy indexing routine:
    for c_score, indices in zip(corner_scores, corner_indices):
        indices = np.array(indices)  # Convert to NumPy array for efficient indexing
        y_coords, x_coords = indices[:, 0], indices[:, 1]

        # Compute pairwise squared distances
        dy = y_coords[:, None] - y_coords[None, :]
        dx = x_coords[:, None] - x_coords[None, :]
        pairwise_dist_sq = dx ** 2 + dy ** 2

        # Get pairwise score comparisons (excluding self-comparison)
        score_dominates = c_score[y_coords[:, None], x_coords[:, None]] > c_score[y_coords[None, :], x_coords[None, :]]

        # Set diagonal to False to ignore self-comparison
        np.fill_diagonal(score_dominates, False)

        # Find the minimum squared distance where a higher score exists
        ranking = np.where(score_dominates, pairwise_dist_sq, np.inf).min(axis=1)
        ranking_indices = np.argsort(ranking)

        best_corners = [indices[i] for i in ranking_indices[-num_best_corners:]]
        anms_outputs.append(best_corners)

    if show:
        show_array = []
        for img, corners in zip(images, anms_outputs):
            img = img.copy()
            for (y, x) in corners:
                cv2.circle(img, (x, y), 3, 255, 3)
            show_array.append(img)
        show_helper(show_array, False, 'ANMS Output')
    return anms_outputs


def extract_feature_descriptors(images, corners, method='Harris'):
    """
    Extract Feature Descriptors
    :param images: list of images
    :param corners: list of corner locations stored as (x, y) tuples
    """

    kernelSize = 41
    outputs = []
    grayscales = [cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32) for img in images]
    for img, corner in zip(grayscales, corners):
        img_output = []
        if method != 'Harris':  # Shi-Tomasi
            # Need to reshape and then invert the x and y coordinates to make it
            # compatible with the code so it works for both this and Harris corners
            corner = corner.reshape(-1, 2)
            temp = corner.copy()
            corner[:, 0], corner[:, 1] = temp[:, 1], temp[:, 0]
        for (y, x) in corner:
            patch = cv2.getRectSubPix(img, (kernelSize, kernelSize), (x.astype(np.float32), y.astype(np.float32)))
            blurred = cv2.GaussianBlur(patch, (kernelSize, kernelSize), 0, None, 0)
            subsampled = cv2.resize(blurred, (8, 8))
            standarized = (subsampled - np.mean(subsampled)) / np.std(subsampled)
            reshaped = np.reshape(standarized, (1, 64))
            img_output.append([(y, x), reshaped])
        outputs.append(img_output)
    return outputs


def match_features(img1, img2, descriptors1, descriptors2, ratio_thresh, show: bool = False):
    """
    Match Features
    :param image1: first image
    :param image2: second image
    :param descriptors1: list of descriptors for image1
    :param descriptors2: list of descriptors for image2
    :param ratio_thresh: ratio threshold for feature matching
    :param show: show the feature matching output
    """
    matches = []
    for desc1 in descriptors1:
        best_match = None
        best_distance = np.inf
        second_best_distance = np.inf
        # Find the best match
        for desc2 in descriptors2:
            distance = np.linalg.norm(desc1[1] - desc2[1])
            # If the distance is less than the best distance, update the best distance and second best distance
            if distance < best_distance:
                second_best_distance = best_distance
                best_distance = distance
                best_match = desc2
        if best_distance / second_best_distance < ratio_thresh:
            matches.append([desc1[0], best_match[0]])

    if show:
        img = np.concatenate((img1, img2), axis=1)
        for (y1, x1), (y2, x2) in matches:
            cv2.circle(img, (x1, y1), 3, (0, 0, 255), 2)
            cv2.circle(img, (x2 + img1.shape[1], y2), 3, (0, 0, 255), 2)
            cv2.line(img, (x1, y1), (x2 + img1.shape[1], y2), (255, 0, 0), 2)
        show_helper([img], False)
    return matches


# Note: RANSAC is using a self-implemented homography matrix calculation
# This could be replaced with the cv2.getPerspectiveTransform function
def run_RANSAC(img1, img2, matches, threshold=10, num_iterations=1000, show: bool = False):
    """
    Run RANSAC
    :param matches: list of matches
    :param threshold: threshold for RANSAC
    :param num_iterations: number of iterations for RANSAC
    :param show: show the RANSAC output
    """
    num_features = len(matches)
    best_inliers = None
    best_percent = 0
    for i in range(num_iterations):
        H = calc_homography_matrix(matches)
        inliers = run_ssd_threshhold(matches, threshold, H)
        percent_match = len(inliers) / num_features
        # if a better match is found, update the best match
        if percent_match > best_percent:
            best_percent = percent_match
            best_inliers = inliers
            print(f"Best Percent: {best_percent}")
    H_hat = get_h_hat(best_inliers)

    if show:
        img = np.concatenate((img1, img2), axis=1)
        for (y1, x1), (y2, x2) in best_inliers:
            cv2.circle(img, (x1, y1), 3, (0, 0, 255), 2)
            cv2.circle(img, (x2 + img1.shape[1], y2), 3, (0, 0, 255), 2)
            cv2.line(img, (x1, y1), (x2 + img1.shape[1], y2), (255, 0, 0), 2)
        show_helper([img], False, title="RANSAC Output")



    print("Number of Inliers: ", len(best_inliers))
    print("Number of Features: ", num_features)
    return best_inliers, H_hat


def make_homography_matrix_row(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    # This is a single set of rows of the A matrix
    out = np.array([[x1, y1, 1, 0, 0, 0, -x1 * x2, -y1 * x1, -x2],
                    [0, 0, 0, x1, y1, 1, -x1 * y1, -y1 * y2, -y2]])
    return out


def calc_homography_matrix(sample_set):
    random_samples = np.random.permutation(sample_set)[:4]
    A_matrix = np.vstack([make_homography_matrix_row(p1, p2) for p1, p2 in random_samples])
    # Solve for homography using SVD
    _, _, Vt = np.linalg.svd(A_matrix)
    H = Vt[-1, :]  # Last row of Vt contains the solution
    # Reshape to 3x3
    H = H.reshape(3, 3)
    H /= H[2, 2]
    return H


def run_ssd_threshhold(matches, threshold, h):
    """
    Run SSD Threshold
    :param matches: list of matches
    :param threshold: threshold for SSD
    :param h: homography matrix estimate
    """
    inliers = []
    for p1, p2 in matches:
        p1 = np.array([p1[0], p1[1], 1])
        p2 = np.array([p2[0], p2[1], 1])
        p2_prime = np.dot(h, p1)
        # Normalize to make the 3rd element 1
        if p2_prime[2] == 0:
            p2_prime[2] = 0.00001
        p2_prime /= p2_prime[2]
        if np.linalg.norm(p2_prime - p2) < threshold:
            inliers.append([p1[:2], p2[:2]])
    return inliers


def get_h_hat(inliers):
    """
    Get Homography Matrix based on all the inliers
    :param inliers: list of inliers
    """
    A = []
    for p1, p2 in inliers:
        x1, y1 = p1
        x2, y2 = p2
        A.append([-x1, -y1, -1, 0, 0, 0, x2 * x1, x2 * y1, x2])
        A.append([0, 0, 0, -x1, -y1, -1, y2 * x1, y2 * y1, y2])
    A = np.array(A)
    U, S, Vt = np.linalg.svd(A)
    H = Vt[-1, :].reshape(3, 3)
    H /= H[2, 2]
    return H


def blend_images(img1, img2, inliers, H):
    """
    Blend Images
    :param img1: first image
    :param img2: second image
    :param inliers: matched feature points used to compute the homography
    :param H: homography matrix (img1 → img2)
    :return: Blended stitched image
    """
    # Get image dimensions
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # Compute warped image 1 corners
    img1_corners = np.array([[0, 0], [0, h1], [w1, h1], [w1, 0]], dtype=np.float32).reshape(-1, 1, 2)
    img1_corners_t = cv2.perspectiveTransform(img1_corners, H)

    # Compute warped image bounds
    x_min, y_min = np.int32(img1_corners_t.min(axis=0).ravel())
    x_max, y_max = np.int32(img1_corners_t.max(axis=0).ravel())

    # Compute translation to move images into positive coordinate space
    translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])

    # Warp img1 using homography
    warped_img1 = cv2.warpPerspective(img1, translation.dot(H), (x_max - x_min, y_max - y_min))

    # Transform inliers for alignment
    img1_inliers = np.array([[p1[1], p1[0]] for p1, p2 in inliers]).reshape(-1, 1, 2).astype(np.float32)
    img1_inliers_t = cv2.perspectiveTransform(img1_inliers, translation.dot(H)).astype(np.int32)
    warped_one_copy = warped_img1.copy()

    # Preview the inliers after homography
    for (x,y) in img1_inliers_t.reshape(-1,2):
        cv2.circle(warped_one_copy, (x, y), 3, (0, 0, 255), 2)
    plt.imshow(warped_one_copy)
    plt.show()
    img2_inliers = np.array(inliers)[:, 1].reshape(-1, 1, 2).astype(np.float32)

    # Compute mean translation based on inliers
    mean_translation = np.mean(img1_inliers_t - img2_inliers, axis=0)

    # Apply translation to img2
    translation_mat = np.array([[1, 0, mean_translation[0][0]], [0, 1, mean_translation[0][1]], [0, 0, 1]])
    warped_img2 = cv2.warpPerspective(img2, translation_mat, (x_max - x_min, y_max - y_min))

    # Create an empty canvas for final blending
    stitched = np.zeros((y_max - y_min, x_max - x_min, 3), dtype=np.uint8)

    # Blend images using alpha blending for smoother transitions
    mask1 = (warped_img1 > 0).astype(np.float32)
    mask2 = (warped_img2 > 0).astype(np.float32)
    overlap = (mask1 + mask2) > 1

    stitched = warped_img1.copy()
    stitched[warped_img2 > 0] = warped_img2[warped_img2 > 0]

    # Blend overlapping areas
    alpha = 0.5
    stitched[overlap] = (alpha * warped_img1[overlap] + (1 - alpha) * warped_img2[overlap]).astype(np.uint8)

    # Display results
    plt.imshow(cv2.cvtColor(stitched, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()

    return stitched


def main():
    # Add any Command Line arguments here
    # Parser = argparse.ArgumentParser()
    # Parser.add_argument('--NumFeatures', default=100, help='Number of best features to extract from each image, Default:100')

    # Args = Parser.parse_args()
    # NumFeatures = Args.NumFeatures

    """
    Read a set of images for Panorama stitching
    """
    images = load_images("../Data/Train", "Set1")
    """
	Corner Detection
	Save Corner detection output as corners.png
	"""
    """
	Perform ANMS: Adaptive Non-Maximal Suppression
	Save ANMS output as anms.png
	"""
    harris_outputs = detect_corners(images, 'Harris', show=True)
    outputs = run_ANMS(images, harris_outputs, 100, 'Harris', show=True)
    # np.save("harris_anms.npy", outputs)

    # shi_tomasi_outputs = detect_corners(images, 'Shi-Tomasi', max_corners=150, show=True)
    # outputs = run_ANMS(images, shi_tomasi_outputs, 100, 'Shi-Tomasi', show=True)
    # np.save("shi_tomasi_anms.npy", outputs)

    """
	Feature Descriptors
	Save Feature Descriptor output as FD.png
	"""
    outputs = np.load("harris_anms.npy", allow_pickle=True)
    descriptors = extract_feature_descriptors(images, outputs)

    # outputs = np.load("shi_tomasi_anms.npy", allow_pickle=True)
    # descriptors = extract_feature_descriptors(images, outputs, 'Shi-Tomasi')

    """
	Feature Matching
	Save Feature Matching output as matching.png
	"""
    matches = match_features(images[0], images[1], descriptors[0], descriptors[1], 0.8, show=True)
    """
	Refine: RANSAC, Estimate Homography
	"""
    best_inliers, H_hat = run_RANSAC(images[0], images[1], matches, 10, 1000, show=True)
    h_cv2 = cv2.findHomography(np.array([i[0] for i in best_inliers]), np.array([i[1] for i in best_inliers]), cv2.RANSAC, 5.0)
    print("CV2 Homography: ", h_cv2[0])
    print("My Homography: ", H_hat)
    print('Difference: ', h_cv2[0] - H_hat)
    """
	Image Warping + Blending
	Save Panorama output as mypano.png
	"""
    blend_images(images[0], images[1], best_inliers, h_cv2[0])


if __name__ == "__main__":
    main()
