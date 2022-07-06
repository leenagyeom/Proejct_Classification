import os
import glob
import cv2
import shutil

from sklearn.model_selection import train_test_split

CATEGORY = {'apple_fuji_l': 0, 'apple_fuji_m': 1, 'apple_fuji_s': 2,
            'apple_yanggwang_l': 3, 'apple_yanggwang_m': 4, 'apple_yanggwang_s': 5,
            'cabbage_green_l': 6, 'cabbage_green_m': 7, 'cabbage_green_s': 8,
            'cabbage_red_l': 9, 'cabbage_red_m': 10, 'cabbage_red_s': 11,
            'chinese-cabbage_l': 12, 'chinese-cabbage_m': 13, 'chinese-cabbage_s': 14,
            'garlic_uiseong_l': 15, 'garlic_uiseong_m': 16, 'garlic_uiseong_s': 17,
            'mandarine_hallabong_l': 18, 'mandarine_hallabong_m': 19, 'mandarine_hallabong_s': 20,
            'mandarine_onjumilgam_l': 21, 'mandarine_onjumilgam_m': 22, 'mandarine_onjumilgam_s': 23,
            'onion_red_l': 24, 'onion_red_m': 25, 'onion_red_s': 26,
            'onion_white_l': 27, 'onion_white_m': 28, 'onion_white_s': 29,
            'pear_chuhwang_l': 30, 'pear_chuhwang_m': 31, 'pear_chuhwang_s': 32,
            'pear_singo_l': 33, 'pear_singo_m': 34, 'pear_singo_s': 35,
            'persimmon_bansi_l': 36, 'persimmon_bansi_m': 37, 'persimmon_bansi_s': 38,
            'persimmon_booyu_l': 39, 'persimmon_booyu_m': 40, 'persimmon_booyu_s': 41,
            'persimmon_daebong_l': 42, 'persimmon_daebong_m': 43, 'persimmon_daebong_s': 44,
            'potato_seolbong_l': 45, 'potato_seolbong_m': 46, 'potato_seolbong_s': 47,
            'potato_sumi_l': 48, 'potato_sumi_m': 49, 'potato_sumi_s': 50,
            'radish_winter-radish_l': 51, 'radish_winter-radish_m': 52, 'radish_winter-radish_s': 53}

def split(path):
    img_path = sorted(glob.glob(os.path.join(path, "raw", "*.png")))
    lab_path = sorted(glob.glob(os.path.join(path, "label", "*.json")))
    train_data = []
    test_data = []

    for cate in CATEGORY.keys():
        img_list = []
        jsn_list = []
        for (img, lab) in zip(img_path, lab_path):
            if cate in img:
                img_list.append(img)
                jsn_list.append(lab)

        x_train, x_test, y_train, y_test = train_test_split(img_list, jsn_list, test_size=0.2, random_state=777)
        train_data.append((x_train, y_train))
        test_data.append((x_test, y_test))

        for (i, j) in zip(x_train, y_train):
            os.makedirs(f"./dataset/train/{cate}", exist_ok=True)
            open_img = cv2.imread(i)
            img_file_name = i.split('\\')[-1]
            jsn_file_name = j.split('\\')[-1]
            cv2.imwrite(f"./dataset/train/{cate}/{img_file_name}", open_img)
            shutil.copy(j, f"./dataset/train/{cate}/{jsn_file_name}")

        for (x, y) in zip(x_test, y_test):
            os.makedirs(f"./dataset/test/{cate}", exist_ok=True)
            open_img = cv2.imread(x)
            img_file_name = x.split('\\')[-1]
            jsn_file_name = y.split('\\')[-1]
            cv2.imwrite(f"./dataset/test/{cate}/{img_file_name}", open_img)
            shutil.copy(y, f"./dataset/test/{cate}/{jsn_file_name}")