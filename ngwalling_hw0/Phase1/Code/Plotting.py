import os

import matplotlib.pyplot as plt

# Most of this is generated with ChatGPT since plotting is a very simple yet tedious task
import matplotlib.pyplot as plt
import cv2


def plot_feature_maps(base_path, image_number):
    """
    Plots BrightnessMap, ColorMap, and TextonMap for a given image number in a single row.

    Parameters:
        image_number (int): The image number to load and display.
    """
    # Define file names
    brightness_map_file = f"{base_path}/Img{image_number}/BrightnessMap_{image_number}.png"
    color_map_file = f"{base_path}/Img{image_number}/ColorMap_{image_number}.png"
    texton_map_file = f"{base_path}/Img{image_number}/TextonMap_{image_number}.png"

    output_location = f"{base_path}/plots"

    file_names = [brightness_map_file, color_map_file, texton_map_file]
    images = []
    # Load images using OpenCV
    for file in file_names:
        img = cv2.imread(file, cv2.IMREAD_UNCHANGED)  # Load image as-is (preserving bit depth)
        if img is None:
            images.append(None)  # Handle missing images
        elif len(img.shape) == 3:  # Convert BGR to RGB for color images
            images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            images.append(img)
            # Plot images
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    titles = ["Brightness Map", "Color Map", "Texton Map"]

    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img)
        ax.set_title(title)
        ax.axis("off")
    plt.title(f"Feature Maps for Image {image_number}")
    plt.tight_layout()
    plt.show()
    plt.imsave(f"{output_location}/FeatureMaps_{image_number}.png", images)


def plot_gradient_maps(base_path, image_number):
    """
    Plots BrightnessMap, ColorMap, and TextonMap for a given image number in a single row.

    Parameters:
        image_number (int): The image number to load and display.
    """
    # Define file names
    brightness_map_file = f"{base_path}/Img{image_number}/Bg_{image_number}.png"
    color_map_file = f"{base_path}/Img{image_number}/Cg_{image_number}.png"
    texton_map_file = f"{base_path}/Img{image_number}/Tg_{image_number}.png"

    output_location = f"{base_path}/plots"

    file_names = [brightness_map_file, color_map_file, texton_map_file]
    images = []
    # Load images using OpenCV
    for file in file_names:
        img = cv2.imread(file, cv2.IMREAD_UNCHANGED)  # Load image as-is (preserving bit depth)
        if img is None:
            images.append(None)  # Handle missing images
        elif len(img.shape) == 3:  # Convert BGR to RGB for color images
            images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            images.append(img)
            # Plot images
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    titles = ["Brightness Gradient Map", "Color Gradient Map", "Texton Gradient Map"]

    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img)
        ax.set_title(title)
        ax.axis("off")
    plt.title(f"Feature Maps for Image {image_number}")
    plt.tight_layout()
    plt.show()
    plt.imsave(f"{output_location}/GradientMaps_{image_number}.png", images)

def plot_image_ground_pblite():
    output_base_path = "Outputs/PbLite/"
    data_base_path = "../BSDS500/"
    raw_images = data_base_path + "Images/"
    ground_truth_images = data_base_path + "GroundTruth/"

    titles = ["Image", "Ground Truth", "PbLite Output"]

    for i in range(1,11):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        raw = cv2.imread(raw_images + f"{i}.jpg", cv2.IMREAD_UNCHANGED)
        ground_truth = cv2.imread(ground_truth_images + f"{i}.png", cv2.IMREAD_UNCHANGED)
        pblite = cv2.imread(output_base_path + f"PbLite_{i}.png", cv2.IMREAD_UNCHANGED)

        raw = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        ground_truth = cv2.cvtColor(ground_truth, cv2.COLOR_BGR2RGB)
        pblite = cv2.cvtColor(pblite, cv2.COLOR_BGR2RGB)

        images = [raw, ground_truth, pblite]
        for ax, img, title in zip(axes, images, titles):
            ax.imshow(img)
            ax.set_title(title)
            ax.axis("off")
        plt.title(f"Feature Maps for Image {i}")
        plt.tight_layout()
        plt.show()
        plt.imsave(f"Outputs/Comparisons/Comparison_{i}.png", images)



def plot_all():
    base_path = "Outputs/"
    plot_image_ground_pblite()
    # Define image number
    for i in range(10):
        #Plot feature maps
        plot_feature_maps(base_path, i + 1)
        plot_gradient_maps(base_path, i + 1)



if __name__ == "__main__":
    plot_all()