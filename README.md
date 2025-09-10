# traffic-sign-detection

German traffic sign detection  with YOLOv8 and recognizion CNN

.

├── CNN.py                 # Training & inference for TSR (ResNet18)

├── dataprocGTSRB.py       # Preprocesses GTSRB dataset for classification

├── dataprocSIGN.py        # Preprocesses GTSDB dataset for detection

├── SIGN.yaml              # Dataset config file for YOLOv8 detection

├── main.py (your script)  # Runs data prep, training, and testing

├── Dataset/

│   ├── GTSRB_Org/         # Original GTSRB dataset (downloaded manually)

│   ├── GTSDB_Org/         # Original GTSDB dataset (downloaded manually)

│   ├── GTSRB/             # Preprocessed GTSRB dataset

│   └── GTSDB/             # Preprocessed GTSDB dataset

Requirements:

Python 3.8+

PyTorch + Torchvision

Ultralytics YOLOv8

OpenCV

Pillow

scikit-learn

pandas

1. GTSRB (Classification)

Download GTSRB dataset: https://benchmark.ini.rub.de/gtsrb_dataset.html

Place it under Dataset/GTSRB_Org/Training/.


3. GTSDB (Detection)

Download GTSDB dataset: https://benchmark.ini.rub.de/gtsdb_dataset.html

Place it under Dataset/GTSDB_Org/FullIJCNN2013/.

The trained models will be saved as:

G1_TSD.pt → YOLO detection model.

CNNGTSRB.pth → ResNet classification model.
