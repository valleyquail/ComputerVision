#!/usr/bin/env python3

"""
RBE/CS549 Spring 2022: Computer Vision
Homework 0: Alohomora: Phase 1 Starter Code

Colab file can be found at:
    https://colab.research.google.com/drive/1FUByhYCYAfpl8J9VxMQ1DcfITpY8qgsF

Author(s):
Prof. Nitin J. Sanket (nsanket@wpi.edu), Lening Li (lli4@wpi.edu), Gejji, Vaishnavi Vivek (vgejji@wpi.edu)
Robotics Engineering Department,
Worcester Polytechnic Institute

Code adapted from CMSC733 at the University of Maryland, College Park.
"""
import os
from os.path import exists

import matplotlib.colors
# Code starts here:

import numpy as np
import cv2
from IPython.core.pylabtools import figsize
from scipy.ndimage import rotate
import matplotlib.pyplot as plt

from Phase1.Code.Plotting import plot_all


####################################################
# Difference of Gaussian Filter Bank
####################################################

def make_gaussian(size, sigmaX=0, sigmaY=0):
    """
    Generate Gaussian filter kernel
    """
    if sigmaX == 0:
        sigmaX = 0.3 * ((size - 1) * 0.5 - 1) + 0.8
    if sigmaY == 0:
        sigmaY = 0.3 * ((size - 1) * 0.5 - 1) + 0.8
    if size % 2 == 0:
        print("Size should be odd, adding 1")
        size += 1
    x_grid, y_grid = np.mgrid[-size / 2 + 0.5:size / 2 + 0.5, -size / 2 + 0.5:size / 2 + 0.5]
    kernel = (np.exp(-((x_grid ** 2) / (2 * sigmaX ** 2) + (y_grid ** 2) / (2 * sigmaY ** 2))))

    return kernel / (2 * np.pi * sigmaX * sigmaY)


def make_DoG(size, sigma_X, sigma_Y, orientation, order=1):
    """
    Generate Difference of Gaussian filter bank
    """
    sobel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

    gaussian = make_gaussian(size, sigma_X, sigma_Y)
    convolved = cv2.filter2D(gaussian, -1, sobel)
    # Works to approximate the second order derivative
    if order == 2:
        convolved = cv2.filter2D(convolved, -1, sobel)
    rotated = rotate(convolved, orientation, reshape=False)

    return rotated


def plot_DoG(kernel_size=63, scales=(4, 8), orientations=16, show=False):
    """
    Make all the filters in the DoG filter bank
    @param scales: list of kernel sigmas
    @param orientations: number orientations
    """
    filters = []
    fig = None
    axs = None
    if show:
        fig, axs = plt.subplots(len(scales), orientations,figsize=(10, len(scales) * 1.2))

    for i, scale in enumerate(scales):
        for orientation in range(orientations):
            kernel = make_DoG(kernel_size, scale, scale, orientation * 360 / orientations)
            filters.append(kernel)
            if show:
                axs[i, orientation].imshow(kernel, cmap='gray')
                axs[i, orientation].axis('off')
    if show:
        fig.tight_layout(pad=0.)  # Reduce spacing
        plt.subplots_adjust(wspace=0.05, hspace=0.0)  # Further tighten layout
        plt.show()
    return filters


####################################################
# Leung-Malik Filter
####################################################

def make_leung_malik(size=63, sigmas=(1, np.sqrt(2), 2, 2 * np.sqrt(2)), orientations=6, show=False):
    """
    Generate Leung-Malik filter bank
    """
    # 48 filters in total
    filters = []
    fig = None
    axs = None
    if show:
        fig, axs = plt.subplots(4, 12, figsize=(10, 4 * 1.2))

    for i, sigma in enumerate(sigmas[0:3]):
        for j in range(orientations):
            # Plots the first order derivatives
            kernel = make_DoG(size, sigma, 3 * sigma, j * 180 / orientations)
            filters.append(kernel)
            if show:
                axs[i, j].imshow(kernel, cmap='gray')
                axs[i, j].axis('off')
            # Plots the second order derivatives
            kernel = make_DoG(size, sigma, 3 * sigma, j * 180 / orientations, 2)
            filters.append(kernel)
            if show:
                axs[i, j + orientations].imshow(kernel, cmap='gray')
                axs[i, j + orientations].axis('off')

    # Lapalcian of Gaussian
    # Approximate Laplacian filter kernel
    laplacian_op = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    for i, sigma in enumerate(sigmas):
        # Basic Gaussian
        kernel = make_gaussian(size, sigma, sigma)
        filters.append(kernel)
        if show:
            axs[3, i + 8].imshow(kernel, cmap='gray')
            axs[3, i + 8].axis('off')

        # No scale factor
        kernel = cv2.filter2D(kernel, -1, laplacian_op)
        filters.append(kernel)
        if show:
            axs[3, i].imshow(kernel, cmap='gray')
            axs[3, i].axis('off')

        # Scale factor of 3
        sigma *= 3
        kernel = make_gaussian(size, sigma, sigma)
        kernel = cv2.filter2D(kernel, -1, laplacian_op)
        filters.append(kernel)
        if show:
            axs[3, i + 4].imshow(kernel, cmap='gray')
            axs[3, i + 4].axis('off')
    if show:
        fig.tight_layout(pad=0)  # Reduce spacing
        plt.subplots_adjust(wspace=0.05, hspace=0.)  # Further tighten layout
        plt.show()
    return filters


####################################################
# Gabor Filter
####################################################

def make_garbor(size, sigma, theta, lambd):
    """
    Generate Gabor filter kernel
    """
    if size % 2 == 0:
        print("Size should be odd, adding 1")
        size += 1
    x_grid, y_grid = np.mgrid[-size / 2 + 0.5:size / 2 + 0.5, -size / 2 + 0.5:size / 2 + 0.5]
    x_prime = x_grid * np.cos(theta) + y_grid * np.sin(theta)
    y_prime = -x_grid * np.sin(theta) + y_grid * np.cos(theta)
    kernel = np.exp(-(x_prime ** 2 + y_prime ** 2) / (2 * sigma ** 2)) * np.cos(2 * np.pi * x_prime / lambd)

    return kernel


def plot_garbor(size=63, sigmas=(4, 8, 12, 20, 25), orientations=8, show=False):
    """
    Generate Gabor filter bank
    """
    fig = None
    axs = None
    if show:
        fig, axs = plt.subplots(len(sigmas), orientations, figsize=(10, len(sigmas) * 1.2))
    filters = []
    for i, sigma in enumerate(sigmas):
        for k in range(orientations):
            lambd = np.pi * sigma / 4
            # The np.pi since the function takes radians
            kernel = make_garbor(size, sigma, k * np.pi / orientations, lambd)
            if show:
                axs[i, k].imshow(kernel, cmap='gray')
                axs[i, k].axis('off')
                filters.append(kernel)
    if show:
        fig.tight_layout(pad=0.05)  # Reduce spacing
        plt.subplots_adjust(wspace=0.05, hspace=0.05)  # Further tighten layout
        plt.show()
    return filters


####################################################
# Image Loader
####################################################

def load_images(path, show=False):
    """
    Load images from path
    """
    images = []
    for img in sorted(os.listdir(path)):
        image = cv2.imread(os.path.join(path, img))
        if img.endswith(".jpg"):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(image)
    if show:
        fig, axs = plt.subplots(1, len(images), figsize=(12, 12))
        for i, image in enumerate(images):
            axs[i].imshow(image)
            axs[i].axis('off')
        plt.show()
    return images


####################################################
# Texton, Brightness, Color Maps
####################################################
def generate_texton_map(image, filters, show=False):
    """
    Generate texton map
    """
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    texton_map = np.zeros((image.shape[0], image.shape[1], len(filters)))
    for i, filter in enumerate(filters):
        filter = filter.astype(np.float32)
        texton_map[:, :, i] = cv2.filter2D(src=image, ddepth=-1, kernel=filter)

    flattened = texton_map.reshape((-1, len(filters))).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, label, _ = cv2.kmeans(flattened, 64, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    texton = np.reshape(label, (image.shape[0], image.shape[1]))

    if show:
        plt.imshow(texton, cmap='viridis')
        plt.title("Texton Map")
        plt.axis('off')
        plt.show()
    return texton.astype(np.uint8)


def generate_brightness_map(image, show=False):
    """
    Generate brightness map
    """
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    flattened = np.reshape(image, (-1, 1)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, label, _ = cv2.kmeans(flattened, 16, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    brightness_map = label.reshape(image.shape[0], image.shape[1])

    if show:
        plt.imshow(brightness_map, cmap='viridis')
        plt.title("Brightness Map")
        plt.axis('off')
        plt.show()
    return brightness_map.astype(np.uint8)


def generate_color_map(image, show=False):
    """
    Generate color map
    """
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    flattened = np.reshape(image, (-1, 3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, label, _ = cv2.kmeans(flattened, 16, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    color_map = label.reshape(image.shape[0], image.shape[1])
    if show:
        plt.imshow(color_map, cmap='viridis')
        plt.title("Color Map")
        plt.axis('off')
        plt.show()
    return color_map.astype(np.uint8)


####################################################
# Half-disk masks
####################################################

def make_half_disk(size, show=False):
    """
    Generate Half-disk masks in 16 orientations
    size should be even for simplicity
    """
    x, y = np.ogrid[-size / 2:size / 2 + 1, -size / 2:size / 2 + 1]
    mask = x ** 2 + y ** 2 <= (size / 2) ** 2 + 1
    mask = mask.astype(np.uint8)
    mask[:size // 2 + 1, :] *= 255
    masks = []
    for i in range(16):
        kernel = rotate(mask, i * 360 / 16, reshape=False)

        kernel = np.where(kernel > 20, 255, 0)
        masks.append(kernel)
    if show:
        fig, axs = plt.subplots(2, 8, figsize=(12, 3))
        for i in range(8):
            axs[0, i].imshow(masks[i], cmap='gray')
            axs[0, i].axis('off')
            axs[1, i].imshow(masks[i + 8], cmap='gray')
            axs[1, i].axis('off')
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.05, hspace=0.05)  # Further tighten layout
        plt.show()
    return masks


def chi_squared(image, half_disks, num_bins, name, show=False):
    """
    Calculate chi-squared distance
    """
    chi_squared_dist = np.zeros((image.shape[0], image.shape[1], len(half_disks) // 2))
    for disk_index in range(0, len(half_disks) // 2, 2):
        for i in range(1, num_bins):
            temp = np.where(image == i, 1, 0).astype(np.float32)
            g_i = cv2.filter2D(temp, -1, half_disks[disk_index])
            h_i = cv2.filter2D(temp, -1, half_disks[disk_index + 1])
            chi_squared_dist[:, :, disk_index // 2] += (g_i - h_i) ** 2 / (2 * (g_i + h_i + 1e-10))

    gradient = np.mean(chi_squared_dist, axis=2)
    if show:
        plt.imshow(gradient, cmap='viridis')
        plt.axis('off')
        plt.title(name)
        plt.show()
    return gradient.astype(np.uint8)


def pb_lite(sobel_baseline, canny_baseline, Tg, Bg, Cg):
    """
    Combine responses to get pb-lite output
    """

    output = np.multiply(np.mean(np.stack((Tg, Bg, Cg), axis=2), axis=2),
                         (0.5 * sobel_baseline + 0.5 * canny_baseline))
    return output


save_all = True
show_all = False
show_filters = True


def main():
    """
    Generate Difference of Gaussian Filter Bank: (DoG)
    Display all the filters in this filter bank and save image as DoG.png,
    use command "cv2.imwrite(...)"
    """
    dog_filters = plot_DoG(show=show_filters)
    """
    Generate Leung-Malik Filter Bank: (LM)
    Display all the filters in this filter bank and save image as LM.png,
    use command "cv2.imwrite(...)"
    """
    lm_filters_small = make_leung_malik(size=49, show=show_filters)
    lm_filters_large = make_leung_malik(size=49, sigmas=(np.sqrt(2), 2, 2 * np.sqrt(2), 4), show=show_filters)
    """
    Generate Gabor Filter Bank: (Gabor)
    Display all the filters in this filter bank and save image as Gabor.png,
    use command "cv2.imwrite(...)"
    """
    garbor_filters = plot_garbor(size=63, sigmas=(2, 4, 6), show=show_filters)
    """
    Generate Half-disk masks
    Display all the Half-disk masks and save image as HDMasks.png,
    use command "cv2.imwrite(...)"
    """
    half_disks_1 = make_half_disk(8, True)
    half_disks_2 = make_half_disk(13, True)
    half_disks_3 = make_half_disk(21, True)
    """
    Load images
    Combine filters into mega list
    """
    imgs = load_images("../BSDS500/Images", show_all)
    names = os.listdir("../BSDS500/Images")
    names = [name.rsplit('.', 1)[0] for name in names]
    names = sorted(names)

    for name in names:
        os.makedirs(f"Outputs/Img{name}", exist_ok=True)
    os.makedirs("Outputs/PbLite", exist_ok=True)
    os.makedirs("RawArrays", exist_ok=True)

    filters = dog_filters + lm_filters_small + lm_filters_large + garbor_filters
    half_disks = half_disks_1 + half_disks_2 + half_disks_3
    """
    Generate Texton Map
    Filter image using oriented gaussian filter bank
    """
    """
    Generate texture ID's using K-means clustering
    Display texton map and save image as TextonMap_ImageName.png,
    use command "cv2.imwrite('...)"
    """
    """
    Generate Texton Gradient (Tg)
    Perform Chi-square calculation on Texton Map
    Display Tg and save image as Tg_ImageName.png,
    use command "cv2.imwrite(...)"
    """
    """
    Generate Brightness Map
    Perform brightness binning 
    """
    """
    Generate Brightness Gradient (Bg)
    Perform Chi-square calculation on Brightness Map
    Display Bg and save image as Bg_ImageName.png,
    use command "cv2.imwrite(...)"
    """
    """
    Generate Color Map
    Perform color binning or clustering
    """
    """
    Generate Color Gradient (Cg)
    Perform Chi-square calculation on Color Map
    Display Cg and save image as Cg_ImageName.png,
    use command "cv2.imwrite(...)"
    """
    if save_all:
        for name, img in zip(names, imgs):
            print("Processing Image", name)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            texton = generate_texton_map(img, filters, show=show_all)
            plt.title("Texton Map")
            plt.imsave(f"Outputs/Img{name}/TextonMap_{name}.png", texton, cmap='viridis')
            np.save(f"RawArrays/TextonMap_{name}.npy", texton)

            brightness = generate_brightness_map(img, show=show_all)
            plt.title("Brightness Map")
            plt.imsave(f"Outputs/Img{name}/BrightnessMap_{name}.png", brightness, cmap='viridis')
            np.save(f"RawArrays/BrightnessMap_{name}.npy", brightness)

            color = generate_color_map(img, show=show_all)
            plt.title("Color Map")
            plt.imsave(f"Outputs/Img{name}/ColorMap_{name}.png", color, cmap='viridis')
            np.save(f"RawArrays/ColorMap_{name}.npy", color)

            texton = np.load(f"RawArrays/TextonMap_{name}.npy")
            brightness = np.load(f"RawArrays/BrightnessMap_{name}.npy")
            color = np.load(f"RawArrays/ColorMap_{name}.npy")

            gradient_tg = chi_squared(texton, half_disks, 64, "Texton Gradient", show=show_all)
            gradient_bg = chi_squared(brightness, half_disks, 16, "Brightness Gradient", show=show_all)
            gradient_cg = chi_squared(color, half_disks, 16, "Color Gradient", show=show_all)

            plt.title("Texton Gradient")
            plt.imsave(f"Outputs/Img{name}/Tg_{name}.png", gradient_tg, cmap='viridis')
            np.save(f"RawArrays/Tg_{name}.npy", gradient_tg)

            plt.title("Brightness Gradient")
            plt.imsave(f"Outputs/Img{name}/Bg_{name}.png", gradient_bg, cmap='viridis')
            np.save(f"RawArrays/Bg_{name}.npy", gradient_bg)

            plt.title("Color Gradient")
            plt.imsave(f"Outputs/Img{name}/Cg_{name}.png", gradient_cg, cmap='viridis')
            np.save(f"RawArrays/Cg_{name}.npy", gradient_cg)

    """
    Read Sobel Baseline
    use command "cv2.imread(...)"
    """
    sobels = load_images("../BSDS500/SobelBaseline", True)
    """
    Read Canny Baseline
    use command "cv2.imread(...)"
    """
    cannys = load_images("../BSDS500/CannyBaseline", True)

    """
    Combine responses to get pb-lite output
    Display PbLite and save image as PbLite_ImageName.png
    use command "cv2.imwrite(...)"
    """

    for name, sobel, canny in zip(names, sobels, cannys):
        print("Processing PB Lite for ", name)
        sobel = cv2.cvtColor(sobel, cv2.COLOR_RGB2GRAY)
        plt.imshow(sobel, cmap='gray')
        plt.title("Sobel Baseline")
        plt.axis('off')
        plt.show()

        canny = cv2.cvtColor(canny, cv2.COLOR_RGB2GRAY)
        plt.imshow(canny, cmap='gray')
        plt.axis('off')
        plt.title("Canny Baseline")
        plt.show()

        gradient_tg = np.load(f"RawArrays/Tg_{name}.npy")
        gradient_bg = np.load(f"RawArrays/Bg_{name}.npy")
        gradient_cg = np.load(f"RawArrays/Cg_{name}.npy")

        pb_lite_output = pb_lite(sobel, canny, gradient_tg, gradient_bg, gradient_cg) * 1.5
        plt.imshow(pb_lite_output, cmap='gray')
        plt.axis('off')
        plt.title("PbLite")
        plt.show()
        # cv2.imwrite(f"Outputs/PbLite/PbLite_{name}.png", pb_lite_output)
        # break


if __name__ == '__main__':
    main()
    plot_all()
