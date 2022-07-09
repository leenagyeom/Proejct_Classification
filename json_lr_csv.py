import os
import glob
import json
from natsort import natsort
import pandas as pd

# image_list = natsort.natsorted(glob.glob(os.path.join("./dataset/","*","*","*.png")))

label_list = natsort.natsorted(glob.glob(os.path.join("./dataset/","*","*","*.json")))
# file = label_list[0]

nm_list = []
lb_list = []
wi_list = []
he_list = []
we_list = []

def take_label(json_file):
    for file in json_file:
        with open(file, "r", encoding='utf-8') as f:
            data = json.load(f)

            name = file.split('\\')[-2]
            nm_list.append(name)
            x_data = data['repo']
            lb_list.append(x_data)

            width = float(data['width'])
            wi_list.append(width)
            height = float(data['height'])
            he_list.append(height)
            weight = float(data['weight'])
            we_list.append(weight)
            # print(width, height, weight)


    label_df = pd.DataFrame({
        'file' : nm_list,
        'x' : lb_list,
        'width' : wi_list,
        'height' : he_list,
        'weight' : we_list
    })

    label_df.to_csv('./linear_train.csv', encoding='utf-8-sig')


if __name__ == "__main__":
    take_label(label_list)

# 여기서부터 y값을 리스트로 묶으니까 학습시킬 때 불러오는 과정에서 문자로 인식하는 경우가 발생.
# 컬럼을 따로 줘서 학습할 때 리스트화 해서 돌리는 게 맞는 것 같다.