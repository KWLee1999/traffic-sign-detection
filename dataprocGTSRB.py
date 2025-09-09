import os
import pandas as pd
import shutil

from PIL import Image
from sklearn.model_selection import train_test_split

def data_processing(src_dir):
    tr_img_dir = 'Dataset/GTSRB/training/images'
    tr_lbl_dir = 'Dataset/GTSRB/training/labels'
    ts_img_dir = 'Dataset/GTSRB/test/images'
    ts_lbl_dir = 'Dataset/GTSRB/test/labels'
    vd_img_dir = 'Dataset/GTSRB/validate/images'
    vd_lbl_dir = 'Dataset/GTSRB/validate/labels'

    ds_dir = [tr_lbl_dir, tr_img_dir, ts_img_dir, ts_lbl_dir, vd_img_dir, vd_lbl_dir]
    for tmp_dir in ds_dir:
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
        os.makedirs(tmp_dir)

    for dir in os.listdir(src_dir):
        if dir == '.DS_Store':
            continue
        inner_dir = os.path.join(src_dir, dir)
        print(inner_dir[-5:])
        if os.path.isdir(inner_dir) == False:
            continue
        for img in os.listdir(inner_dir):
            if img == "GT-" + dir + '.csv':
                csv_file = pd.read_csv(os.path.join(inner_dir,"GT-" + dir + '.csv'), sep=';')
                csv_file['Filename'] = csv_file['Filename'].apply(lambda x: f'{inner_dir[-5:]}_{x}')
                csv_file.to_csv(f'{inner_dir}/GGT-{dir}.csv', sep=';', index=False)
            else:
                continue
                # os.rename(inner_dir + '/' + img, inner_dir + '/' + f'{inner_dir[-5:]}_{img}')

    for dir in os.listdir(src_dir):
        if dir == '.DS_Store' or dir == 'images' or dir == 'labels':
            continue
        inner_dir = os.path.join(src_dir, dir)
        print(inner_dir[-5:])
        if os.path.isdir(inner_dir) == False:
            continue
        for img in os.listdir(inner_dir):
            if img == "GT-" + dir + '.csv' or img == "GGT-" + dir + '.csv' or img == '00000_GT-00000.gsheet': # omit all files except the images
                continue
            else:
                shutil.copy(f'{inner_dir}/{img}', tr_img_dir)
                os.rename(tr_img_dir + '/' + img, tr_img_dir + '/' + f'{inner_dir[-5:]}_{img}')

    train_csv = pd.DataFrame()

    for dir in os.listdir(src_dir):
        if dir == '.DS_Store':
            continue

        inner_dir = os.path.join(src_dir, dir)
        csv_file_path = os.path.join(inner_dir, "GGT-" + dir + '.csv')

        if os.path.exists(csv_file_path):
            csv_file = pd.read_csv(csv_file_path, sep=';')
            train_csv = pd.concat([train_csv, csv_file], ignore_index=True)

    print(train_csv.shape)

    def compare(v1, v2):
        if v1 > v2:
            vmax, vmin = v1, v2
            return vmax, vmin
        else:
            vmax, vmin = v2, v1
            return vmax, vmin


    def convert_labels(z):
        x1 = z['Roi.X1']
        y1 = z['Roi.Y1']
        x2 = z['Roi.X2']
        y2 = z['Roi.Y2']
        size = [z['Height'],z['Width']]
        xmax, xmin = compare(x1, x2)
        ymax, ymin = compare(y1, y2)
        dw = 1./size[1]
        dh = 1./size[0]
        x = (xmin + xmax)/2.0
        y = (ymin + ymax)/2.0
        w = xmax - xmin
        h = ymax - ymin
        x = x*dw
        w = w*dw
        y = y*dh
        h = h*dh
        return x, y, w, h


    for index, x in train_csv.iterrows():
        name = x['Filename'].replace('.ppm', '.txt')
        labels = convert_labels(x)
        cls = x['ClassId']
        f = open(f'{tr_lbl_dir}/{name}', 'w+')
        f.write(f'{cls} {labels[0]} {labels[1]} {labels[2]} {labels[3]}')
        f.close()

    y = train_csv['ClassId'] ## For stratification

    train_ds, valid_ds = train_test_split(
        train_csv,
        test_size=0.2,
        random_state=43,
        shuffle=True,
        stratify=y
    )

    y = valid_ds['ClassId']

    train_ds.reset_index()
    valid_ds.reset_index()

    f = open(f'Dataset/GTSRB/train.txt', 'w+')
    for index, x in train_ds.iterrows():
        name = x['Filename']
        f.write(f'{tr_img_dir}/{name}\n')
    f.close()

    f = open(f'Dataset/GTSRB/valid.txt', 'w+')
    for index, x in valid_ds.iterrows():
        img_name = x['Filename']
        lbl_name = x['Filename'].replace('.ppm', '.txt')
        f.write(f'{tr_lbl_dir}/{img_name}\n')
        shutil.move(f'{tr_img_dir}/{img_name}', f'{vd_img_dir}/{img_name}')
        shutil.move(f'{tr_lbl_dir}/{lbl_name}', f'{vd_lbl_dir}/{lbl_name}')
    f.close()

    classes = [
        'Speed limit (20km/h)',
        'Speed limit (30km/h)',
        'Speed limit (50km/h)',
        'Speed limit (60km/h)',
        'Speed limit (70km/h)',
        'Speed limit (80km/h)',
        'End of speed limit (80km/h)',
        'Speed limit (100km/h)',
        'Speed limit (120km/h)',
        'No passing',
        'No passing veh over 3.5 tons ',
        'Right-of-way at intersection ',
        'Priority road ',
        'Yield ',
        'Stop ',
        'No vehicles ',
        'Veh > 3.5 tons prohibited ',
        'No entry ',
        'General caution ',
        'Dangerous curve left ',
        'Dangerous curve right ',
        'Double curve ',
        'Bumpy road ',
        'Slippery road ',
        'Road narrows on the right ',
        'Road work ',
        'Traffic signals ',
        'Pedestrians ',
        'Children crossing ',
        'Bicycles crossing ',
        'Beware of ice/snow',
        'Wild animals crossing ',
        'End speed + passing limits ',
        'Turn right ahead ',
        'Turn left ahead ',
        'Ahead only ',
        'Go straight or right ',
        'Go straight or left ',
        'Keep right ',
        'Keep left ',
        'Roundabout mandatory ',
        'End of no passing ',
        'End no passing veh > 3.5 tons'
    ]

    f = open(f'Dataset/GTSRB/classes.names', 'w+')
    for c in classes:
        f.write(f'{c}\n')
    f.close()

    ## labelled_data.data ##

    config = {
        'classes': 43,
        'train': 'Dataset/GTSRB/train.txt',
        'valid': 'Dataset/GTSRB/valid.txt',
        'test': 'Dataset/GTSRB/test.txt',
        'names': 'classes.names',
        'backup': 'backup'
    }
    f = open(f'Dataset/GTSRB/labelled_data.data', 'w+')
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

