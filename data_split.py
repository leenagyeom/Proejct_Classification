import os
import glob
import cv2
import shutil

from sklearn.model_selection import train_test_split

CATEGORY = {"apple_fuji":0, "apple_yanggwang":1, "cabbage_green":2, "cabbage_red":3, "chinese-cabbage":4,
            "garlic_uiseong":5, "mandarine_hallabong":6, "mandarine_onjumilgam":7, "onion_red":8, "onion_white":9,
            "pear_chuhwang":10, "pear_singo":11, "persimmon_bansi":12, "persimmon_booyu":13, "persimmon_daebong":14,
            "potato_seolbong":15, "potato_sumi":16, "radish_winter-radish":17}

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