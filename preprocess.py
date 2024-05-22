import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def create_subfolders(base_dir, subfolders):
    for subfolder in subfolders:
        folder_path = os.path.join(base_dir, subfolder)
        os.makedirs(folder_path, exist_ok=True)

def load_image_paths_and_labels(folder, label):
    image_paths = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        image_paths.append(img_path)
        labels.append(label)
    return image_paths, labels

def rotate_and_save_images(image_paths, labels, save_dir, image_size=(128, 128)):
    for img_path, label in tqdm(zip(image_paths, labels), total=len(image_paths)):
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, image_size)
            for angle in [0, 90, 180, 270]:
                rotated_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE if angle == 90 else (cv2.ROTATE_180 if angle == 180 else (cv2.ROTATE_90_COUNTERCLOCKWISE if angle == 270 else None)))
                label_dir = 'Fake' if label == 1 else 'Real'
                save_path = os.path.join(save_dir, label_dir, f"{os.path.splitext(os.path.basename(img_path))[0]}_{angle}.png")
                cv2.imwrite(save_path, rotated_img)
        else:
            print(f"Warning: Image at path {img_path} could not be read.")

def preprocess_images(ai_generated_dir, real_images_dir, train_dir, test_dir, test_size=0.2):
    ai_image_paths, ai_labels = load_image_paths_and_labels(ai_generated_dir, 1)
    real_image_paths, real_labels = load_image_paths_and_labels(real_images_dir, 0)

    image_paths = ai_image_paths + real_image_paths
    labels = ai_labels + real_labels

    X_train_paths, X_test_paths, y_train, y_test = train_test_split(image_paths, labels, test_size=test_size, random_state=42)

    create_subfolders(train_dir, ['Fake', 'Real'])
    create_subfolders(test_dir, ['Fake', 'Real'])

    print("Processing training images...")
    rotate_and_save_images(X_train_paths, y_train, train_dir)
    print("Processing testing images...")
    rotate_and_save_images(X_test_paths, y_test, test_dir)

def load_preprocessed_data(train_dir, test_dir):
    train_image_paths = []
    train_labels = []
    for label, subfolder in enumerate(['Fake', 'Real']):
        folder_path = os.path.join(train_dir, subfolder)
        for filename in os.listdir(folder_path):
            train_image_paths.append(os.path.join(folder_path, filename))
            train_labels.append(label)

    test_image_paths = []
    test_labels = []
    for label, subfolder in enumerate(['Fake', 'Real']):
        folder_path = os.path.join(test_dir, subfolder)
        for filename in os.listdir(folder_path):
            test_image_paths.append(os.path.join(folder_path, filename))
            test_labels.append(label)

    return train_image_paths, train_labels, test_image_paths, test_labels
