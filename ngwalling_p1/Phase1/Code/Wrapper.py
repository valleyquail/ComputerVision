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
import argparse

# Code starts here:

import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from skimage import exposure


def show_helper(inputs, tiled=True, title=None, save=False):
    if tiled:
        fig, axes = plt.subplots(1, len(inputs))
        for i, img in enumerate(inputs):
            axes[i].imshow(img)
            axes[i].axis('off')
            if title:
                axes[i].set_title(title)
        plt.tight_layout()
        if save:
            plt.savefig(os.path.join(SavePath, title) + '.png')
        plt.show()
    else:
        for i, img in enumerate(inputs):
            plt.imshow(img)
            plt.axis('off')
            if title:
                plt.title(title)
            if save:
                plt.savefig(os.path.join(SavePath, title) + str(i) + '.png')
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


def detect_corners(images, method: str, max_corners=100, show: bool = False, save: bool = True):
    """
    Corner Detection
    :param images: list of images
    :param method: corner detection method: either 'Harris' or 'Shi-Tomasi'
    :param show: show the corner detection output
    """
    outputs = []
    # convert to grayscale
    grayscales = [cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) for img in images]
    for img in grayscales:
        _, mask = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)
        if method == 'Harris':
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                # Get the largest contour (assumes black border is the outermost area)
                largest_contour = max(contours, key=cv2.contourArea)
                mask = np.zeros_like(img)
                cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
            else:
                mask = np.ones_like(img) * 255
                # Apply the mask to the image
            gray_masked = cv2.bitwise_and(img, img, mask=mask)
            corners = cv2.cornerHarris(gray_masked, 4, 3, 0.04)
            outputs.append(corners)
        elif method == 'Shi-Tomasi':
            corners = cv2.goodFeaturesToTrack(img, maxCorners=max_corners, qualityLevel=0.01, minDistance=30, mask=mask, blockSize=7)
            corners = np.int32(corners)
            outputs.append(corners)
        else:
            raise ValueError("Invalid corner detection method")
    if show or save:
        corners = []
        for img, dst in zip(images, outputs):
            # Make a copy of the image as to not modify the original
            img = img.copy()
            if 'Shi-Tomasi' == method:
                for i in dst:
                    x, y = i.ravel()
                    cv2.circle(img, (x, y), 3, 255, 3)
                corners.append(img)
            else:
                img[dst > 0.01 * dst.max()] = [255, 0, 0]
                corners.append(img)
        show_helper(corners, False, 'corners', save=save)
    return outputs


def run_ANMS(images, corner_scores, num_best_corners, method: str, show: bool = False, save=False):
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
            corner_indices.append(np.argwhere(score > 0.005 * score.max()))
    elif method == 'Shi-Tomasi':
        print("Shi-Tomasi already does ANMS")
        return corner_scores
    else:
        raise ValueError("Invalid corner detection method")

    # Run ANMS
    anms_outputs = []
    ## This is the original code that uses nested for loops naively
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

    if show or save:
        show_array = []
        for img, corners in zip(images, anms_outputs):
            img = img.copy()
            for (y, x) in corners:
                cv2.circle(img, (x, y), 3, 255, 3)
            show_array.append(img)
        show_helper(show_array, False, 'anms', save=save)
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


def match_features(img1, img2, descriptors1, descriptors2, ratio_thresh, show: bool = False, save=False):
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

    if show or save:
        # Create cv2.KeyPoint objects for descriptors
        keypoints1 = [cv2.KeyPoint(p[1].astype(np.float32), p[0].astype(np.float32), 1) for p in
                      np.array(descriptors1)[:, 0].tolist()]
        keypoints2 = [cv2.KeyPoint(p[1].astype(np.float32), p[0].astype(np.float32), 1) for p in
                      np.array(descriptors2)[:, 0].tolist()]
        matchescv = [cv2.DMatch(i, i, 0) for i in range(len(matches))]
        # Draw the matches
        img_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matchescv, None, matchesThickness=2,
                                      matchColor=(0, 255, 0), singlePointColor=(0, 0, 255),
                                      flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        # img = np.concatenate((img1, img2), axis=1)
        # for (y1, x1), (y2, x2) in matches:
        #     cv2.circle(img, (x1, y1), 3, (0, 0, 255), 2)
        #     cv2.circle(img, (x2 + img1.shape[1], y2), 3, (0, 0, 255), 2)
        #     cv2.line(img, (x1, y1), (x2 + img1.shape[1], y2), (255, 0, 0), 2)
        show_helper([img_matches], False, 'matching', save=save)
    return matches


# Note: RANSAC is using a self-implemented homography matrix calculation
# This could be replaced with the cv2.getPerspectiveTransform function
def run_RANSAC(img1, img2, matches, threshold=10, num_iterations=2000, show: bool = False, save=False):
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
        inliers1 = [cv2.KeyPoint(float(p1[1]), float(p1[0]), 1) for p1, p2 in best_inliers]
        inliers2 = [cv2.KeyPoint(float(p2[1]), float(p2[0]), 1) for p1, p2 in best_inliers]
        matchescv = [cv2.DMatch(i, i, 0) for i in range(len(best_inliers))]
        img_matches = cv2.drawMatches(img1, inliers1, img2, inliers2, matchescv, None, matchesThickness=2,
                                      matchColor=(0, 255, 0), singlePointColor=(0, 0, 255),
                                      flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        show_helper([img_matches], False, title="RANSAC", save=save)

    print("Number of Inliers: ", len(best_inliers))
    print("Number of Features: ", num_features)
    return best_inliers, H_hat


def make_homography_matrix_row(p1, p2):
    y1, x1 = p1
    y2, x2 = p2
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
        p1_temp = np.array([p1[1], p1[0], 1])
        p2_temp = np.array([p2[1], p2[0], 1])
        p2_prime = np.dot(h, p1_temp)
        # Normalize to make the 3rd element 1
        if p2_prime[2] == 0:
            p2_prime[2] = 0.00001
        p2_prime /= p2_prime[2]
        if np.linalg.norm(p2_prime - p2_temp) < threshold:
            inliers.append([p1, p2])
    return inliers


def get_h_hat(inliers):
    """
    Get Homography Matrix based on all the inliers
    :param inliers: list of inliers
    """
    A = []
    for p1, p2 in inliers:
        y1, x1 = p1
        y2, x2 = p2
        A.append([-x1, -y1, -1, 0, 0, 0, x2 * x1, x2 * y1, x2])
        A.append([0, 0, 0, -x1, -y1, -1, y2 * x1, y2 * y1, y2])
    A = np.array(A)
    U, S, Vt = np.linalg.svd(A)
    H = Vt[-1, :].reshape(3, 3)
    H /= H[2, 2]
    return H


def blend_images(img1, img2, inliers, H, mode='alpha'):
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

    # Compute warped image 1 corners
    img1_corners = np.array([[0, 0], [0, h1], [w1, h1], [w1, 0]], dtype=np.float32).reshape(-1, 1, 2)
    img1_corners_t = cv2.perspectiveTransform(img1_corners, H)
    # Compute warped image bounds
    x_min, y_min = np.floor(img1_corners_t.min(axis=0).ravel()).astype(np.int32)
    x_max, y_max = np.floor(img1_corners_t.max(axis=0).ravel()).astype(np.int32)

    # Compute translation to move images into positive coordinate space
    translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]]).dot(H)
    x_size = x_max - x_min + w1
    y_size = y_max - y_min + h1
    # Warp img1 using homography
    warped_img1 = cv2.warpPerspective(img1, translation, (x_size, y_size))

    # Transform inliers for alignment
    img1_inliers = np.array([[p1[1], p1[0]] for p1, p2 in inliers]).reshape(-1, 1, 2).astype(np.float32)
    img1_inliers_t = cv2.perspectiveTransform(img1_inliers, translation).reshape(-1, 2)

    # Preview the inliers after homography
    warped_one_copy = warped_img1.copy()
    for (x, y) in img1_inliers_t:
        cv2.circle(warped_one_copy, (x.astype(np.int32), y.astype(np.int32)), 3, (0, 0, 255), 2)
    plt.imshow(warped_one_copy)
    plt.title("Inliers_after_Homography")
    plt.show()

    img2_inliers = np.array([[p2[1], p2[0]] for p1, p2 in inliers]).reshape(-1, 2).astype(np.float32)

    # Compute mean translation based on inliers
    mean_translation_img1 = np.mean(img1_inliers_t - img2_inliers, axis=0).reshape(-1)
    mean_translation_img2 = np.zeros(2)
    # Deal with translations and clipping the image in case some of the movements are negative
    # If one of the movements are negative and applied to image one, set the translation to zero for image one
    # and then make image two move in the opposite direction
    if mean_translation_img1[0] < 0:
        mean_translation_img2[0] = -mean_translation_img1[0]
        mean_translation_img1[0] = 0
    if mean_translation_img1[1] < 0:
        mean_translation_img2[1] = -mean_translation_img1[1]
        mean_translation_img1[1] = 0
    # Apply translation to img2
    translation_mat_img2 = np.array([[1, 0, mean_translation_img1[0]], [0, 1, mean_translation_img1[1]], [0, 0, 1]])
    translation_mat_img1 = np.array([[1, 0, mean_translation_img2[0]], [0, 1, mean_translation_img2[1]], [0, 0, 1]])

    # img2 = perform_color_correction(img2, img1, inliers)

    warped_img2 = cv2.warpPerspective(img2, translation_mat_img2, (x_size, y_size))
    warped_img1 = cv2.warpPerspective(warped_img1, translation_mat_img1, (x_size, y_size))

    mask2 = (warped_img2 > 0).astype(np.uint8)
    stitched = warped_img1.copy()
    stitched[warped_img2 > 0] = warped_img2[warped_img2 > 0]

    if mode == 'alpha':
        # Blend images using alpha blending for smoother transitions
        mask1 = (warped_img1 > 0).astype(np.uint8)
        overlap = (mask1 + mask2) > 1
        alpha = 0.5
        stitched[overlap] = (alpha * warped_img1[overlap] + (1 - alpha) * warped_img2[overlap]).astype(np.uint8)
        output = stitched
    else:
        # Blend images using seamless cloning/poisson blending
        temp = remove_black_space(stitched)
        center = (temp.shape[1] // 2, temp.shape[0] // 2)
        output = cv2.seamlessClone(warped_img2, stitched, mask2, center, cv2.MIXED_CLONE)
    blended_img = remove_black_space(output)

    return blended_img


def perform_color_correction(img1, img2, inliers):
    """
    Perform Color Correction
    :param img1: first image --> image to be color corrected
    :param img2: second image --> reference image
    :param inliers: the matched feature points used to compute the homography
    Converts the color space of the first image to match the second image.
    """
    kernelSize = 41
    mean_colors = np.zeros(3)  # For mean color difference across all inliers

    for (y1, x1), (y2, x2) in inliers:
        # Get small patches around the matched points (inliers) from both images
        patch1 = cv2.getRectSubPix(img1, (kernelSize, kernelSize), (x1.astype(np.float32), y1.astype(np.float32)))
        patch2 = cv2.getRectSubPix(img2, (kernelSize, kernelSize), (x2.astype(np.float32), y2.astype(np.float32)))

        # Apply Gaussian blur to patches to reduce noise
        blurred1 = cv2.GaussianBlur(patch1, (kernelSize, kernelSize), 0)
        blurred2 = cv2.GaussianBlur(patch2, (kernelSize, kernelSize), 0)

        # Subsample both patches to 8x8, then reshape to a flat array of color values
        subsampled1 = cv2.resize(blurred1, (8, 8)).reshape(-1, 3)
        subsampled2 = cv2.resize(blurred2, (8, 8)).reshape(-1, 3)

        # Calculate the mean color difference between the two patches
        mean_colors += np.mean(subsampled1, axis=0) - np.mean(subsampled2, axis=0)

    # Average the color differences over all inliers
    mean_colors /= len(inliers)

    mask = img1 > 1  # Boolean mask for non-black pixels
    # Apply mean color correction only where mask is True
    img1_corrected = img1.astype(np.float32)
    img1_corrected += mask * mean_colors
    # Clip and convert back to uint8
    img1_corrected = np.clip(img1_corrected, 0, 255).astype(np.uint8)

    return img1_corrected


def remove_black_space(img):
    """
    Remove the black space (background) around an image by cropping the non-black content.
    :param img: Input image (with black space to be removed)
    :return: Cropped image without the black space
    """
    y_nonzero, x_nonzero, _ = np.nonzero(img)
    return img[np.min(y_nonzero):np.max(y_nonzero), np.min(x_nonzero):np.max(x_nonzero)]


def make_panorama(images, numFeatures, threshold, iterations, method, mode, show=True, save=False):
    base_image = images[0]
    # Queue to keep track of images to be stitched
    image_queue = [[0, img] for img in images[1:]]
    skipped = []
    while len(image_queue) > 0:
        img1 = base_image
        count, img2 = image_queue.pop(0)
        images = [img2, img1]
        if method == 'Harris':
            harris_outputs = detect_corners([img2, img1], 'Harris', show=True, save=save)
            outputs = run_ANMS(images, harris_outputs, numFeatures + 10 * count, 'Harris', show=True, save=save)
            descriptors = extract_feature_descriptors(images, outputs)
        else:
            outputs = detect_corners([img2, img1], 'Shi-Tomasi', max_corners=numFeatures, show=True,
                                                save=save)
            descriptors = extract_feature_descriptors(images, outputs, 'Shi-Tomasi')

        matches = match_features(images[0], images[1], descriptors[0], descriptors[1], 0.8, show=True)
        if len(matches) < 20:
            print("Not enough matches found, skipping this image")
            count += 1
            if count < 5:
                image_queue.append([count, img2])
            else:
                skipped.append(img2)
                print("Too many failed attempts for this image, skipping")
            continue
        best_inliers, H_hat = run_RANSAC(images[0], images[1], matches, threshold, iterations, True)
        if len(best_inliers) < 5:
            print("Not enough percentage of inliers, skipping this image")
            count += 1
            if count < 5:
                image_queue.append([count, img2])
            else:
                skipped.append(img2)
                print("Too many failed attempts for this image, skipping")
            continue
        base_image = blend_images(images[0], images[1], best_inliers, H_hat, mode=mode)
        # Increase the number of features since the image is now larger
        numFeatures += 50
        # Make sure it only saves once
        save = False
    if show:
        if len(skipped) > 0:
            show_helper(skipped, True, 'Skipped Images')
        plt.imshow(base_image)
        plt.axis('off')
        plt.title('Panorama')
        plt.imsave(os.path.join(SavePath, 'mypano.png'), base_image)
        plt.show()
    return base_image


def stitch_pair(img1, img2, numFeatures, threshold, iterations, method, show):
    images = [img1, img2]
    if method == 'Harris':
        harris_outputs = detect_corners(images, 'Harris', show=True)
        outputs = run_ANMS(images, harris_outputs, numFeatures, 'Harris', show=True)
        descriptors = extract_feature_descriptors(images, outputs)
    else:
        shi_tomasi_outputs = detect_corners(images, 'Shi-Tomasi', max_corners=numFeatures + 50, show=True)
        outputs = run_ANMS(images, shi_tomasi_outputs, numFeatures, 'Shi-Tomasi', show=True)
        descriptors = extract_feature_descriptors(images, outputs, 'Shi-Tomasi')

    matches = match_features(images[0], images[1], descriptors[0], descriptors[1], 0.6, show=True)

    best_inliers, H_hat = run_RANSAC(images[0], images[1], matches, threshold, iterations, show=True)

    # Verify the homography matrix using cv2.findHomography as a reference
    # h_cv2 = cv2.findHomography(np.array([i[0] for i in best_inliers]), np.array([i[1] for i in best_inliers]),
    #                            cv2.RANSAC, 5.0)
    # print("CV2 Homography: ", h_cv2[0])
    # print("My Homography: ", H_hat)
    # print('Difference: ', h_cv2[0] - H_hat)
    # print("Normalized Difference: ", np.linalg.norm(h_cv2[0] - H_hat))
    # print("Determinant My: ", np.linalg.det(H_hat))
    # print("Determinant CV2: ", np.linalg.det(h_cv2[0]))
    # print("Determinant Difference: ", np.linalg.det(h_cv2[0]) - np.linalg.det(H_hat))

    blend_images(images[0], images[1], best_inliers, H_hat)


global SavePath


def main():
    # Add any Command Line arguments here
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--Path', default='../Data/Train', help='Path to the dataset, Default: ../Data/Train')
    Parser.add_argument('--NumFeatures', default=100,
                        help='Number of best features to extract from each image, Default:100')
    Parser.add_argument('--Set', default='Set3', help='Dataset to use, Default: Set1')
    Parser.add_argument('--Method', default='Shi-Tomasi", Default: Shi-Tomasi')
    Parser.add_argument('--Iterations', default=3000, help='Number of iterations for RANSAC, Default: 2000')
    Parser.add_argument('--Threshold', default=15, help='Threshold for RANSAC, Default: 10')
    Parser.add_argument('--Mode', default='alpha', help='Blending mode to use, Default: alpha')
    Parser.add_argument('--ColorCorrection', default=True, help='Perform color correction, Default: True')
    Parser.add_argument('--SavePath', default='../Data/Results',
                        help='Path to save the output, Default: ../Data/Results')
    Parser.add_argument('--Save', default=True, help='Save the output, Default: True')
    Parser.add_argument('--Show', default=True, help='Show the output, Default: True')

    Args = Parser.parse_args()
    Path = Args.Path
    Set = Args.Set
    Save = Args.Save
    global SavePath
    SavePath = os.path.join(Args.SavePath, Set)

    NumFeatures = Args.NumFeatures
    Method = Args.Method
    Show = Args.Show
    Threshold = Args.Threshold
    Iterations = Args.Iterations

    """
    Read a set of images for Panorama stitching
    """
    images = load_images(Path, Set, show=Show)
    os.makedirs(SavePath, exist_ok=True)

    # A bunch of stuff for saving images:
    # harris_corners = detect_corners(images, 'Harris', show=Show, save=Save)
    # anms = run_ANMS(images, harris_corners, NumFeatures, 'Harris', show=Show, save=Save)
    # descriptors = extract_feature_descriptors(images, anms)
    # matches = match_features(images[0], images[1], descriptors[0], descriptors[1], 0.7, show=Show, save=Save)

    # best_inliers, H_hat = run_RANSAC(images[0], images[1], matches, Threshold, Iterations, show=Show, save=Save)

    # stitch_pair(images[0], images[1], NumFeatures, Threshold, Iterations, Method, Show)
    make_panorama(images, NumFeatures, Threshold, Iterations, Method, Args.Mode, True, Save)

    """
	Corner Detection
	Save Corner detection output as corners.png
	"""

    """
	Perform ANMS: Adaptive Non-Maximal Suppression
	Save ANMS output as anms.png
	"""
    """
    Feature Descriptors
    Save Feature Descriptor output as FD.png
    """
    """
	Feature Matching
	Save Feature Matching output as matching.png
	"""
    """
	Refine: RANSAC, Estimate Homography
	"""
    """
	Image Warping + Blending
	Save Panorama output as mypano.png
	"""


if __name__ == "__main__":
    main()
