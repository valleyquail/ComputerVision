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
from skimage.feature import peak_local_max
import os
import matplotlib.pyplot as plt


def show_helper(inputs, tiled=True):
    if tiled:
        fig, axes = plt.subplots(1, len(inputs))
        for i, img in enumerate(inputs):
            axes[i].imshow(img)
            axes[i].axis('off')
        plt.show()
    else:
        for img in inputs:
            plt.imshow(img)
            plt.axis('off')
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


def detect_corners(images, method: str, show: bool = False):
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
            corners = cv2.cornerHarris(img, 3, 3, 0.04)
            outputs.append(corners)
    elif method == 'Shi-Tomasi':
        for img in grayscales:
            corners = cv2.goodFeaturesToTrack(img, 1000, 0.01, 0.04)
            outputs.append(corners)
    else:
        raise ValueError("Invalid corner detection method")
    if show:
        corners = []
        for img, dst in zip(images, outputs):
            # Make a copy of the image as to not modify the original
            img = img.copy()
            if 'Shi-Tomasi' == method:
                dst = cv2.dilate(dst, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
                dst = dst.astype(np.int32)
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
    for c_score, indices in zip(corner_scores, corner_indices):
        ranking = np.inf * np.ones(len(indices))
        # Reverse the order of the indices since i is for rows (y) and j is for columns (x)
        for i, (y_i, x_i) in enumerate(indices):
            for j, (y_j, x_j) in enumerate(indices):
                if i == j:
                    continue
                if c_score[y_j, x_j] > c_score[y_i, x_i]:
                    ranking[i] = min(ranking[i], np.power(x_j - x_i, 2) + np.power(y_j - y_i, 2))
        ranking_indices = np.argsort(ranking)
        best_corners = [indices[i] for i in ranking_indices[-num_best_corners:]]
        anms_outputs.append(best_corners)
    if show:
        show_array = []
        for img, corners in zip(images, anms_outputs):
            for (y, x) in corners:
                cv2.circle(img, (x, y), 3, 255, 3)
            show_array.append(img)
        show_helper(show_array, False)
    return anms_outputs


def extract_feature_descriptors(images, corners):
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
        for desc2 in descriptors2:
            distance = np.linalg.norm(desc1[1] - desc2[1])
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
def run_RANSAC(img1, img2, matches, threshold=10, num_iterations=100, show: bool = False):
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
        if percent_match > best_percent:
            best_percent = percent_match
            best_inliers = inliers
    H_hat = calc_homography_matrix(best_inliers)
    if show:
        img = np.concatenate((img1, img2), axis=1)
        for (x1, y1), (x2, y2) in best_inliers:
            cv2.circle(img, (x1, y1), 3, (0, 0, 255), 2)
            cv2.circle(img, (x2 + img1.shape[1], y2), 3, (0, 0, 255), 2)
            cv2.line(img, (x1, y1), (x2 + img1.shape[1], y2), (255, 0, 0), 2)
        show_helper([img], False)

    return best_inliers, H_hat


def make_homography_matrix_row(p1, p2):
    y1, x1 = p1
    y2, x2 = p2
    out = np.array([[x1, y1, 1, 0, 0, 0, -x1 * x2, -y1 * x1],
                    [0, 0, 0, x1, y1, 1, -x1 * y1, -y1 * y2]])
    return out


def calc_homography_matrix(sample_set):
    random_samples = np.random.permutation(sample_set)[:4]

    A_matrix = np.vstack([make_homography_matrix_row(p1, p2) for p1, p2 in random_samples])
    B_matrix = np.hstack([np.array([p2[1], p2[0]]) for p1, p2 in random_samples]).T

    H = np.hstack([np.linalg.lstsq(A_matrix, B_matrix, rcond=None)[0], 1])
    H = H.reshape(3, 3)
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
        p1 = np.array([p1[1], p1[0], 1])
        p2 = np.array([p2[1], p2[0], 1])
        p2_prime = np.dot(h, p1)
        # Normalize to make the 3rd element 1
        p2_prime /= p2_prime[2]
        if np.linalg.norm(p2_prime - p2) < threshold:
            inliers.append([p1[:-1], p2[:-1]])
    return inliers

def blend_images(img1, img2, inliers, H):
    """
    Blend Images
    :param img1: first image
    :param img2: second image
    :param H: homography matrix
    """
    # Assume image one is being applied to image two
    # Start by applying the homography matrix to the image
    warped = cv2.warpPerspective(img1, H, (img1.shape[1] + img2.shape[1], img1.shape[0]))
    # Warp the key points to match the homography
    img1_points = [p1 for p1, p2 in inliers]
    points = np.array(img1_points, dtype=np.float32)
    points = np.vstack([points.T, np.ones(len(points))]).T
    points_warped = np.dot(H, points.T).T.astype(np.int32)

    for i, point in enumerate(points_warped):
        cv2.circle(warped, (point[0], point[1]), 3, (0, 0, 255), 2)

    # Calculate the mean vector of the points to identify the required translation
    img2_points = [p2 for p1, p2 in inliers]
    x, y = 0, 0
    for i, point2 in enumerate(img2_points):
        x += point2[0] - points_warped[i][0]
        y += point2[1] - points_warped[i][1]
    x /= len(img2_points)
    y /= len(img2_points)





    plt.imshow(warped)
    plt.show()


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

    harris_outputs = detect_corners(images, 'Harris', show=True)
    # shi_tomasi_outputs = detect_corners(images, 'Shi-Tomasi', show=True)


    """
	Perform ANMS: Adaptive Non-Maximal Suppression
	Save ANMS output as anms.png
	"""
    # outputs = run_ANMS(images, harris_outputs, 100, 'Harris', show=True)
    # np.save("harris_anms.npy", outputs)

    """
	Feature Descriptors
	Save Feature Descriptor output as FD.png
	"""
    outputs = np.load("harris_anms.npy", allow_pickle=True)
    descriptors = extract_feature_descriptors(images, outputs)
    """
	Feature Matching
	Save Feature Matching output as matching.png
	"""
    matches = match_features(images[0], images[1], descriptors[0], descriptors[1], 0.8, show=True)
    """
	Refine: RANSAC, Estimate Homography
	"""
    best_inliers, H_hat = run_RANSAC(images[0], images[1], matches, 10, 100, show=True)
    """
	Image Warping + Blending
	Save Panorama output as mypano.png
	"""
    blend_images(images[0], images[1], best_inliers, H_hat)

if __name__ == "__main__":
    main()
