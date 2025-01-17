import cv2
import matplotlib.pyplot as plt
import numpy as np

from Wrapper import *

imgs = load_images("../BSDS500/Images")
cannys = load_images("../BSDS500/CannyBaseline")
sobels = load_images("../BSDS500/SobelBaseline")


dog_filters = plot_DoG(show=True)
lm_filters = make_leung_malik(size=49, show=True) + make_leung_malik(size=49,sigmas=(np.sqrt(2),2,2*np.sqrt(2),4), show=True)
garbor_filters = plot_garbor(size=63, sigmas=(2, 4, 6), show=True)

filters = dog_filters + lm_filters + garbor_filters

texton = generate_texton_map(imgs[1], filters, show=True)
#
brightness = generate_brightness_map(imgs[1], show=True)
color = generate_color_map(imgs[1], show=True)

# # texton = np.load("RawArrays/texton_img9.npy")
# # brightness = np.load("RawArrays/brightness_img9.npy")
# # color = np.load("RawArrays/color_img9.npy")
#
half_disks_1 = make_half_disk(8, True)
half_disks_2 = make_half_disk(13, True)
half_disks_3 = make_half_disk(21, True)
# #
gradient_tg = chi_squared(texton, half_disks_1+ half_disks_2 + half_disks_3, 64,"Texton", show=True)
gradient_bg = chi_squared(brightness, half_disks_1+ half_disks_2 + half_disks_3, 16, "Brightness", show=True)
gradient_cg = chi_squared(color, half_disks_1+ half_disks_2 + half_disks_3, 16, "Color", show=True)
#
sobel = cv2.cvtColor(sobels[1], cv2.COLOR_BGR2GRAY)
canny = cv2.cvtColor(cannys[1], cv2.COLOR_BGR2GRAY)

pb_lite_output = pb_lite(sobel, canny, gradient_tg, gradient_bg, gradient_cg)

plt.imshow(pb_lite_output, cmap='gray')
plt.show()