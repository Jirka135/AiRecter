import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from keras.utils import Sequence

def create_subfolders(base_dir, subfolders):
    """Creates subdirectories within a base directory."""
    for subfolder in subfolders:
        folder_path = os.path.join(base_dir, subfolder)
        os.makedirs(folder_path, exist_ok=True)

def load_image_paths_and_labels(folder, label):
    """Loads image file paths and assigns the given label to each image."""
    image_paths = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        image_paths.append(img_path)
        labels.append(label)
    return image_paths, labels

def rotate_and_save_images(image_paths, labels, save_dir, image_size=(128, 128)):
    """Rotates images and saves them in specified directories."""
    for img_path, label in tqdm(zip(image_paths, labels), total=len(image_paths)):
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, image_size)
            for angle in [0, 90, 180, 270]:
                if angle == 90:
                    rotated_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                elif angle == 180:
                    rotated_img = cv2.rotate(img, cv2.ROTATE_180)
                elif angle == 270:
                    rotated_img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                else:
                    rotated_img = img
                
                label_dir = 'Fake' if label == 1 else 'Real'
                save_path = os.path.join(save_dir, label_dir, f"{os.path.splitext(os.path.basename(img_path))[0]}_{angle}.png")
                cv2.imwrite(save_path, rotated_img)
        else:
            print(f"Warning: Image at path {img_path} could not be read.")

def preprocess_images(ai_generated_dir, real_images_dir, train_dir, test_dir, test_size=0.2):
    """Preprocesses images by loading, splitting, rotating, and saving them."""
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
    """Loads preprocessed images from the train and test directories."""
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

class ImageGenerator(Sequence):
    """Custom data generator for loading images in batches during training."""
    def __init__(self, image_paths, labels, batch_size=32, image_size=(512, 512)):
        self.image_paths = image_paths
        self.labels = labels
        self.batch_size = batch_size
        self.image_size = image_size
        self.on_epoch_end()
    
    def __len__(self):
        return int(np.floor(len(self.image_paths) / self.batch_size))
    
    def __getitem__(self, index):
        batch_image_paths = self.image_paths[index * self.batch_size:(index + 1) * self.batch_size]
        batch_labels = self.labels[index * self.batch_size:(index + 1) * self.batch_size]
        images = []
        valid_labels = []
        for img_path, label in zip(batch_image_paths, batch_labels):
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, self.image_size)
                images.append(img)
                valid_labels.append(label)
            else:
                print(f"Warning: Image at path {img_path} could not be read.")
        
        if len(images) != self.batch_size:
            print(f"Warning: Batch size mismatch. Expected {self.batch_size}, got {len(images)}. Skipping batch.")
            return self.__getitem__((index + 1) % self.__len__())
        
        images = np.array(images) / 255.0
        return np.array(images), np.array(valid_labels)
    
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.image_paths))
        np.random.shuffle(self.indexes)
        self.image_paths = [self.image_paths[i] for i in self.indexes]
        self.labels = [self.labels[i] for i in self.indexes]
