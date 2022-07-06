import os

label_list = os.listdir("./QC_sample/label")
image_list = os.listdir("./QC_sample/raw")

for l, i in zip(label_list, image_list):
    re_l = l.lower()
    re_i = i.lower()
    os.rename(f"./QC_sample/label/{l}", f"./QC_sample/label/{re_l}")
    os.rename(f"./QC_sample/raw/{i}", f"./QC_sample/raw/{re_i}")