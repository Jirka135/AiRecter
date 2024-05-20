import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras._tf_keras.keras.utils import Sequence
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout


def load_image_paths_and_labels(folder, label):
    image_paths = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        image_paths.append(img_path)
        labels.append(label)
    return image_paths, labels

ai_generated_dir = 'C:\\Users\\Jirka\\VScode\\AirRect\\AiRecter\\Images\\Fake'
real_images_dir = 'C:\\Users\\Jirka\\VScode\\AirRect\\AiRecter\\Images\\Real'

ai_image_paths, ai_labels = load_image_paths_and_labels(ai_generated_dir, 1)
real_image_paths, real_labels = load_image_paths_and_labels(real_images_dir, 0)

image_paths = ai_image_paths + real_image_paths
labels = ai_labels + real_labels

X_train_paths, X_test_paths, y_train, y_test = train_test_split(image_paths, labels, test_size=0.2, random_state=42)

class ImageDataGenerator(Sequence):
    def __init__(self, image_paths, labels, batch_size=32, image_size=(128, 128)):
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
        
        # Ensure consistent batch size
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

train_generator = ImageDataGenerator(X_train_paths, y_train, batch_size=32)
test_generator = ImageDataGenerator(X_test_paths, y_test, batch_size=32)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(train_generator, epochs=10, validation_data=test_generator)

loss, accuracy = model.evaluate(test_generator)
print(f'Test Accuracy: {accuracy * 100:.2f}%')

model.save('ai_image_recognition_model.h5')