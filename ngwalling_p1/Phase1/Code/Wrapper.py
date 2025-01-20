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
        np.uint8(image)
        images.append(image)
    if show:
        show_helper(images)

    return images

def detect_corners(images, method: str ,show: bool = False):
    """
    Corner Detection
    :param images: list of images
    :param method: corner detection method: either 'Harris' or 'Shi-Tomasi'
    :param show: show the corner detection output
    """
    outputs = []
    #convert to grayscale
    grayscales = [cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32) for img in images]
    if method == 'Harris':
        for img in grayscales:
            corners = cv2.cornerHarris(img, 4, 9, 0.05)
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
    #Making them into a list of lists to make the for loops not kill themselves
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
            # if ranking[i] == np.inf:
            #     ranking[i] = 0
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
    for img, corner in zip(images, corners):
        img_output = []
        for (y, x) in corner:
            patch = cv2.getRectSubPix(img, (kernelSize, kernelSize), (x, y))
            blurred = cv2.GaussianBlur(patch, (kernelSize, kernelSize), 0, None, 0)
            subsampled = cv2.resize(blurred, (8, 8))
            standarized = (subsampled - np.mean(subsampled)) / np.std(subsampled)
            reshaped = np.reshape(standarized, (1, 64))
            img_output.append([(x,y), reshaped])
        outputs.append(img_output)
    return outputs

def match_features(descriptors1, descriptors2):
    pass


def main():
    # Add any Command Line arguments here
    # Parser = argparse.ArgumentParser()
    # Parser.add_argument('--NumFeatures', default=100, help='Number of best features to extract from each image, Default:100')

    # Args = Parser.parse_args()
    # NumFeatures = Args.NumFeatures

    """
    Read a set of images for Panorama stitching
    """
    images = load_images("../Data/Train", "Set1", show=True)
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
    run_ANMS(images, harris_outputs, 40, 'Harris', show=True)

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
