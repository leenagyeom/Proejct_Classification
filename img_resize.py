import random
import numpy as np
import os
import cv2            # pip install opencv-python
import glob
from PIL import Image # pip install Pillow
import PIL.ImageOps

num_augmented_img = 50

file_list = glob.glob(os.path.join("./Validation","*","*.png"))

# file_path = "./Training/apple_fuji_L_copy/"
# file_name = os.listdir(file_path)

file_name = file_list[0].split("\\")[-1]

total_origin_image_run = len(file_list)
print("total image number >>", total_origin_image_run)


for img_path in file_list:
    image = Image.open(img_path)

    resized_image = image.resize(size=(224, 224))
    resized_image.save(img_path)