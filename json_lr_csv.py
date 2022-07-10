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
on_list = []

def take_label(json_file):
    for file in json_file:
        with open(file, "r", encoding='utf-8') as f:
            data = json.load(f)

            file_name = file.split('\\')[-1]
            bar_point = file_name.rfind('-')
            qc = file_name[:bar_point]

            label = file.split('\\')[-2].split('_')
            label = label[0]+'_'+label[-1]

            x_data = data['repo']
            width = float(data['width'])
            height = float(data['height'])
            weight = float(data['weight'])

            if qc not in on_list:
                on_list.append(qc)
                nm_list.append(label)
                lb_list.append(x_data)
                wi_list.append(width)
                he_list.append(height)
                we_list.append(weight)


    label_df = pd.DataFrame({
        'add_QC' : on_list,
        'label' : nm_list,
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