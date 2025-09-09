# Import necessary libraries
from PIL import Image, ImageOps
import cv2
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report

# unique class
subfolder_path = []
u_class = {}
n_class = 0
# Setup device for PyTorch (use CUDA if available).
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_class = 43

# Read the folder path and classes
tr_img_dir = 'Dataset/GTSRB/training/images'
tr_lbl_dir = 'Dataset/GTSRB/training/labels'
ts_img_dir = 'Dataset/GTSRB/validate/images'
ts_lbl_dir = 'Dataset/GTSRB/validate/labels'

# Define the transformations for preprocessing the images
data_transforms = transforms.Compose([
    transforms.ToPILImage(),  # Convert image to PIL Image format.
    transforms.Resize((224, 224)),  # Resize to fit ResNet18's input size.
    transforms.ToTensor(),
    # Normalize according to values suitable for ResNet18.
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def class_name():
    classnames = []
    f = open(f'Dataset/GTSRB/classes.names')
    lines = f.readlines()
    for line in lines:
        classnames.append(line.strip())
    f.close()

    return classnames

# # Data Augmentation
# data_augmentation = transforms.Compose([
#     transforms.ToPILImage(),  # Convert image to PIL Image format.
#     transforms.Resize((224, 224)),  # Resize to fit ResNet18's input size.
#     # Apply random augmentations to increase dataset variety.
#     transforms.GaussianBlur(kernel_size=(3, 5), sigma=(0.1, 3)),
#     transforms.RandomPerspective(distortion_scale=0.3, p=0.3),
#     transforms.ToTensor(),
#     # Normalize according to values suitable for ResNet18.
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# ])

def train(epoch, model):
    model_name = model.__class__.__name__
    print(f'Model: {model_name}')
    # Freeze all layers in the model. We'll only train the final layer.
    for param in model.layer2.parameters():  # Assuming you are using ResNet and 'layer4' is the last block
        param.requires_grad = True
    
    # Config the model =============================================================
    # Modify the final fully connected layer to classify 43 classes.
    model.fc = nn.Sequential(
        nn.Dropout(0.1),  # Apply dropout before the final layer
        nn.Linear(model.fc.in_features, n_class)
    )
    model = model.to(device)
    
    # Define the loss function and optimizer.
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

    # Lists to hold processed images and their labels.
    tr_data = []
    tr_labels = []
    
    # Read and preprocess images from each class directory.
    for filename in os.listdir(f'{tr_img_dir}'):
        print(filename)
        if filename[-4:] == '.png':
            random_number = random.random()
            img = cv2.imread(f'{tr_img_dir}/{filename}')

            labelfile = filename.replace('.png', '.txt')
            if os.path.exists(f'{tr_lbl_dir}/{labelfile}'):
                f = open(f'{tr_lbl_dir}/{labelfile}', 'r')
                lines = f.readlines()
                for line in lines:
                    var = line.strip().split(' ')
                    index = int(var[0])
                    print(f'{index}')
                    if index != '':
                        img_tsf = data_transforms(img)
                        tr_labels.append(index)
                        tr_data.append(img_tsf)
                        # if random_number <= 0.1:
                        #     img_aug = data_augmentation(img)
                        #     tr_data.append(img_aug)
                        #     tr_labels.append(index)
                        if random_number <= 0.5:
                            # Assuming `img` is a color image (BGR format in OpenCV)
                            img_rbg = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
                            # Equalize the Y (luminance) channel
                            img_rbg[:, :, 0] = cv2.equalizeHist(img_rbg[:, :, 0])
                            # Convert back to BGR
                            img_equal = cv2.cvtColor(img_rbg, cv2.COLOR_YUV2BGR)
                            img_equal = data_transforms(img_equal)
                            tr_data.append(img_equal)
                            tr_labels.append(index)
    ts_data = []
    ts_labels = []

    # Read and preprocess images from each class directory.
    for filename in os.listdir(f'{ts_img_dir}'):
        if filename[-4:] == '.png':
            img = cv2.imread(f'{ts_img_dir}/{filename}')
            img = data_transforms(img)

            labelfile = filename.replace('.png','.txt')

            if os.path.exists(f'{ts_lbl_dir}/{labelfile}'):
                print(f'{ts_lbl_dir}/{labelfile}')
                f = open(f'{ts_lbl_dir}/{labelfile}', 'r')
                lines = f.readlines()
                for line in lines:
                    var = line.strip().split(' ')
                    index = int(var[0])
                    print(index)
                    if index != '':
                        ts_labels.append(index)
                        ts_data.append(img)

    # Convert data and labels to PyTorch tensors.
    data_tensor = torch.stack(tr_data)
    labels_tensor = torch.tensor(tr_labels)
    
    # Create a PyTorch dataset and data loader.
    train_set = TensorDataset(data_tensor, labels_tensor)

    # Convert data and labels to PyTorch tensors.
    data_tensor = torch.stack(ts_data)
    labels_tensor = torch.tensor(ts_labels)

    test_set = TensorDataset(data_tensor, labels_tensor)

    # training =====================================================================
    dataloader = DataLoader(train_set, batch_size=16, shuffle=True)
    
    # Training loop for the model.
    num_epochs = epoch
    for epoch in range(num_epochs):
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()  # Zero out any gradient from the previous step.
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()  # Compute the gradients.
            optimizer.step()  # Update the weights.
    
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")
    
    # Save the model's state dictionary
    model_path = 'CNNGTSRB.pth'
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    # Testing ======================================================================
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode if using for inference
    # Initiate the test set
    testloader = torch.utils.data.DataLoader(test_set, batch_size=16, shuffle=False)
    
    # Initialize variables to track accuracy for each class
    y_test = []
    y_pred = []
    
    # Iterate through the test dataset and calculate accuracy for each class
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
    
            # Calculate accuracy for each class
            y_test.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    
    # Convert the numeric class to charactor
    y_test_txt = []
    y_pred_txt = []
    classnames = class_name()
    for i in y_pred:
        y_pred_txt.append(classnames[i])
    
    for i in y_test:
        y_test_txt.append(classnames[i])
    
    # Print the classification report
    report = classification_report(y_test_txt, y_pred_txt)
    
    print(f'Classification Report of {model_name}:\n{report}')


def pred(frame, model):
    model.eval()

    img_rbg = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    # Equalize the Y (luminance) channel
    img_rbg[:, :, 0] = cv2.equalizeHist(img_rbg[:, :, 0])
    # Convert back to BGR
    img_equal = cv2.cvtColor(img_rbg, cv2.COLOR_YUV2BGR)
    input_img = data_transforms(img_equal).unsqueeze(0).to(device)

    with torch.no_grad():  # It doesn't need to keep track of gradients.
        output = model(input_img)
        # Apply softmax to get the probabilities (confidence levels)
        softmax = nn.Softmax(dim=1)  # Softmax over the class dimension
        probabilities = softmax(output)

        _, pred = torch.max(output, 1)  # get the predicted class

        # Map numeric predictions back to class names.
        confidence_level = probabilities[0, pred].item()
        label = pred.item()
    
    return label, round(confidence_level, 2)

# if __name__ == "__main__":
#     train(epoch=25, model=resnet18(pretrained=True))
#     # model, model_path = train(epoch = 10, model = resnet152(pretrained = True))
#     # liveClassification(model,model_path)
    