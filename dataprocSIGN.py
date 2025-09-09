import os
import pandas as pd
import shutil

from PIL import Image
from sklearn.model_selection import train_test_split

def data_processing(src_dir):
    tr_img_dir = 'Dataset/SIGN/training/images'
    tr_lbl_dir = 'Dataset/SIGN/training/labels'
    ts_img_dir = 'Dataset/SIGN/test/images'
    ts_lbl_dir = 'Dataset/SIGN/test/labels'
    vd_img_dir = 'Dataset/SIGN/validate/images'
    vd_lbl_dir = 'Dataset/SIGN/validate/labels'

    txt_file = os.path.join(f'{src_dir}/gt.txt')
    image_width = 1360
    image_height = 800

    ds_dir = [tr_lbl_dir, tr_img_dir, ts_img_dir, ts_lbl_dir, vd_img_dir, vd_lbl_dir]
    for tmp_dir in ds_dir:
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
        os.makedirs(tmp_dir)

    ds = []
    for dir in os.listdir(src_dir):
        if dir == '.DS_Store' or dir == 'images' or dir == 'labels':
            continue
        inner_dir = os.path.join(src_dir, dir)
        if os.path.isdir(inner_dir) == False:
            # print(inner_dir[-4:])
            if inner_dir[-4:] == '.txt':
                continue
            else:
                ds.append(inner_dir)
                shutil.copy(f'{inner_dir}', tr_img_dir)

    with open(txt_file, 'r') as file:
        lines = file.readlines()

        for line in lines:
            # Split the line by semicolon
            parts = line.strip().split(';')
            img_name, leftCol, topRow, rightCol, bottomRow, class_id = parts

            # Convert to float and calculate normalized values
            leftCol, topRow, rightCol, bottomRow = map(float, [leftCol, topRow, rightCol, bottomRow])
            x_center = (leftCol + rightCol) / 2 / image_width
            y_center = (topRow + bottomRow) / 2 / image_height
            bbox_width = (rightCol - leftCol) / image_width
            bbox_height = (bottomRow - topRow) / image_height

            if 0 <= int(class_id) <= 42:
                class_id = 0 #road sign
            else:
                class_id = 1 #other

            # Prepare YOLO format
            yolo_format = f"{class_id:.6f} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}\n"
            output_file = os.path.join(tr_lbl_dir, img_name.replace('.ppm', '.txt'))
            if os.path.exists(output_file):
                with open(output_file, 'a') as out_file:
                    out_file.write(yolo_format)
            else:
                # Write the YOLOv8 format to a .txt file with the same image name
                with open(output_file, 'w+') as out_file:
                    out_file.write(yolo_format)

    train_ds, valid_ds = train_test_split(
        ds,
        test_size=0.2,
        random_state=43,
        shuffle=True
    )

    f = open(f'Dataset/SIGN/train.txt', 'w+')
    for x in train_ds:
        name = x[-9:]
        f.write(f'{tr_img_dir}/{name}\n')
    f.close()

    f = open(f'Dataset/SIGN/valid.txt', 'w+')
    for x in valid_ds:
        img_name = x[-9:]
        lbl_name = x[-9:].replace('.ppm', '.txt')
        f.write(f'{tr_lbl_dir}/{img_name}\n')
        shutil.move(f'{tr_img_dir}/{img_name}', f'{vd_img_dir}/{img_name}')
        if os.path.exists(f'{tr_lbl_dir}/{lbl_name}'):
            shutil.move(f'{tr_lbl_dir}/{lbl_name}', f'{vd_lbl_dir}/{lbl_name}')
        else:
            label_file = open(f'{vd_lbl_dir}/{lbl_name}', 'w+')
            label_file.close()
    f.close()

    # f = open(f'Dataset/SIGN/test.txt', 'w+')
    # for index, x in test_ds.iterrows():
    #     img_name = x[0]
    #     lbl_name = x[0].replace('.ppm', '.txt')
    #     f.write(f'{tr_lbl_dir}/{img_name}\n')
    #     shutil.move(f'{tr_img_dir}/{img_name}', f'{ts_img_dir}/{img_name}')
    #     shutil.move(f'{tr_lbl_dir}/{lbl_name}', f'{ts_lbl_dir}/{lbl_name}')
    # f.close()

    classes = [
        'Road Sign',
        'Other'
    ]

    f = open(f'Dataset/SIGN/classes.names', 'w+')
    for c in classes:
        f.write(f'{c}\n')
    f.close()

    ## labelled_data.data ##

    config = {
        'classes': 2,
        'train': 'Dataset/SIGN/train.txt',
        'valid': 'Dataset/SIGN/valid.txt',
        'test': 'Dataset/SIGN/test.txt',
        'names': 'classes.names',
        'backup': 'backup'
    }
    f = open(f'Dataset/SIGN/labelled_data.data', 'w+')
    for key in config:
        f.write(f'{key} = {config[key]}\n')
    f.close()

    # Convert ppm to png
    cv_paths = [tr_img_dir, vd_img_dir, ts_img_dir]
    for path in cv_paths:
        for filename in os.listdir(path):
            if filename.endswith('.ppm'):
                # Open the .ppm image
                ppm_image_path = os.path.join(path, filename)
                with Image.open(ppm_image_path) as img:
                    # Convert to .png
                    png_image_path = os.path.join(path, filename.replace('.ppm', '.png'))
                    img.save(png_image_path, 'PNG')
                os.remove(ppm_image_path)
                print(f"Converted {filename} to PNG format.")

