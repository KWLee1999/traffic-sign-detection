
import dataprocGTSRB as dpGTSRB
import dataprocSIGN as dpSIGN
import torch.nn as nn
import CNN
import cv2
import torch
import torch.optim as optim
from ultralytics import YOLO
from torchvision.models import resnet18

# Download dataset from: https://benchmark.ini.rub.de/gtsrb_dataset.html to below path
src_dir_GTSRB = 'Dataset/GTSRB_Org/Training/'
# Download dataset from: https://benchmark.ini.rub.de/gtsdb_dataset.html to below path
src_dir_GTSDB = 'Dataset/GTSDB_Org/FullIJCNN2013/'


def data_processing():
    dpGTSRB.data_processing(src_dir_GTSRB)
    dpSIGN.data_processing(src_dir_GTSDB)


def train_TSD_model(model, yaml, epochs, imgsize, batch, ptoutput):
    # Train the YOLO model
    model.train(data=yaml, epochs=epochs, imgsz=imgsize, batch=batch, device=0, optimizer='AdamW'
                , weight_decay=1e-4)
    model.val(data=yaml, imgsz=imgsize)
    model.save(ptoutput)
    torch.cuda.empty_cache()


def train_TSR_model(model, epochs):
    # Train the model
    CNN.train(epoch=epochs, model=model)
    torch.cuda.empty_cache()


def pred_cam(model):
    # Open the webcam (0 is usually the default webcam; change to 1 if you have multiple cameras)
    cap = cv2.VideoCapture(0)

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()

    # Read frames from the webcam in a loop
    while True:
        ret, frame = cap.read()  # Capture frame-by-frame
        if not ret:
            print("Error: Could not read frame.")
            break

        # Use the YOLOv8 model to predict on the frame
        results = model(frame, data='gtsrb.yaml', stream=True)  # Use stream=True for video/webcam

        # Draw the detection results on the frame
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0]
                # Get confidence score and class index
                conf = box.conf[0]
                class_id = box.cls[0]
                # Draw rectangle and label on the frame
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                label = f"{model.names[int(class_id)]} {conf:.2f}"
                cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Show the frame with detections
        cv2.imshow('YOLOv8 Webcam', frame)

        # Press 'q' to quit the video stream
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close the window
    cap.release()
    cv2.destroyAllWindows()


def pred_video_test(tsdmodel, tsrmodel):
    # Open the video file
    video_path = 'Test.mp4'
    video_capture = cv2.VideoCapture(video_path)

    # Loop through each frame of the video
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break  # Break the loop if no frame is returned (end of video)

        # Run YOLO detection on the frame
        results = tsdmodel(frame, device=0)

        # Extract bounding boxes (in xyxy format) from YOLO results
        bounding_boxes = results[0].boxes.xyxy.cpu().numpy()  # Bounding boxes in (x_min, y_min, x_max, y_max)

        # Loop through each bounding box and draw it on the frame
        for box in bounding_boxes:
            x_min, y_min, x_max, y_max = map(int, box)  # Convert to integers for OpenCV
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(frame.shape[1], x_max)
            y_max = min(frame.shape[0], y_max)

            # Crop the frame using the bounding box coordinates
            cropped_frame = frame[y_min:y_max, x_min:x_max]

            classnames = CNN.class_name()
            result, confid_lvl = CNN.pred(cropped_frame, tsrmodel)

            if confid_lvl > 0.4:
                # Extract class predictions and confidence scores from the second model
                classname = classnames[result]

                # Draw the bounding box (rectangle) on the frame
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color=(0, 255, 0), thickness=2)  # Green box
                cv2.putText(frame, f'{classname}: {confid_lvl}', (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Optional: Display the video with bounding boxes (can slow down the process)
        cv2.imshow('YOLO Video Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break  # Press 'q' to exit the video

    # Release the video capture and writer objects
    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    data_processing()

    # Train
    # Yolo TSD
    TSD_model = YOLO('yolov8n.pt')
    train_TSD_model(TSD_model, yaml='SIGN.yaml', epochs=85, imgsize=640, batch=4, ptoutput='G1_TSD.pt')

    # Prediction
    # Initiate Models
    tsd_model = YOLO('G1_TSD.pt')

    tsr_model = resnet18(pretrained=True)
    tsr_model.fc = nn.Sequential(
        nn.Dropout(0.1),  # Apply dropout before the final layer
        nn.Linear(tsr_model.fc.in_features, 43) #43 classese
    )
    model = tsr_model.to(torch.device("cuda"))

    tsr_model.load_state_dict(torch.load('CNNGTSRB.pth'))

    pred_video_test(tsd_model, tsr_model)
