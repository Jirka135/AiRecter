import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras._tf_keras.keras.utils import Sequence
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from keras._tf_keras.keras.regularizers import l2
from keras._tf_keras.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import classification_report
from sklearn.utils import resample

# Set the number of threads for TensorFlow to use
num_threads = os.cpu_count()
tf.config.threading.set_inter_op_parallelism_threads(num_threads)
tf.config.threading.set_intra_op_parallelism_threads(num_threads)

def load_image_paths_and_labels(folder, label):
    """Load image paths and their corresponding labels from a given folder."""
    image_paths = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        image_paths.append(img_path)
        labels.append(label)
    return image_paths, labels

# Directories for AI-generated and real images
ai_generated_dir = 'C:\\Users\\Jirka\\VScode\\AirRect\\AiRecter\\Images\\Fake'
real_images_dir = 'C:\\Users\\Jirka\\VScode\\AirRect\\AiRecter\\Images\\Real'

# Load image paths and labels
ai_image_paths, ai_labels = load_image_paths_and_labels(ai_generated_dir, 1)
real_image_paths, real_labels = load_image_paths_and_labels(real_images_dir, 0)

# Combine and split data into training and testing sets
image_paths = ai_image_paths + real_image_paths
labels = ai_labels + real_labels
X_train_paths, X_test_paths, y_train, y_test = train_test_split(image_paths, labels, test_size=0.2, random_state=42)

class CustomImageDataGenerator(Sequence):
    def __init__(self, image_paths, labels, batch_size=256, image_size=(128, 128), augment=False, **kwargs):
        super().__init__(**kwargs)  # Ensure proper handling of additional arguments
        self.image_paths = image_paths
        self.labels = labels
        self.batch_size = batch_size
        self.image_size = image_size
        self.augment = augment
        self.datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2
        ) if augment else ImageDataGenerator()
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
                # Log unreadable images
                print(f"Warning: Image at path {img_path} could not be read.")
        
        if len(images) != self.batch_size:
            # Log batch size mismatches
            print(f"Warning: Batch size mismatch. Expected {self.batch_size}, got {len(images)}. Skipping batch.")
            return self.__getitem__((index + 1) % self.__len__())
        
        images = np.array(images) / 255.0
        images, valid_labels = np.array(images), np.array(valid_labels)
        if self.augment:
            return next(self.datagen.flow(images, valid_labels, batch_size=self.batch_size))
        return images, valid_labels
    
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.image_paths))
        np.random.shuffle(self.indexes)
        self.image_paths = [self.image_paths[i] for i in self.indexes]
        self.labels = [self.labels[i] for i in self.indexes]

class BalancedBatchGenerator(Sequence):
    def __init__(self, image_paths, labels, batch_size=512, image_size=(128, 128), augment=True, augment_count=5, **kwargs):
        super().__init__(**kwargs)  # Ensure proper handling of additional arguments
        self.image_paths = image_paths
        self.labels = labels
        self.batch_size = batch_size
        self.image_size = image_size
        self.augment = augment
        self.augment_count = augment_count  # Number of augmentations per image
        self.datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2
        ) if augment else ImageDataGenerator()
        self.on_epoch_end()
    
    def __len__(self):
        return int(np.floor(len(self.image_paths) * (self.augment_count + 1) / self.batch_size))
    
    def __getitem__(self, index):
        batch_image_paths = self.image_paths[index * self.batch_size // (self.augment_count + 1):(index + 1) * self.batch_size // (self.augment_count + 1)]
        batch_labels = self.labels[index * self.batch_size // (self.augment_count + 1):(index + 1) * self.batch_size // (self.augment_count + 1)]
        
        images = []
        labels = []
        for img_path, label in zip(batch_image_paths, batch_labels):
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, self.image_size)
                images.append(img)
                labels.append(label)
                for _ in range(self.augment_count):
                    augmented_img = next(self.datagen.flow(np.expand_dims(img, axis=0), batch_size=1))[0]
                    images.append(augmented_img)
                    labels.append(label)
            else:
                print(f"Warning: Image at path {img_path} could not be read.")

        images = np.array(images) / 255.0
        labels = np.array(labels)
        
        return images, labels
    
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.image_paths))
        np.random.shuffle(self.indexes)
        self.image_paths = [self.image_paths[i] for i in self.indexes]
        self.labels = [self.labels[i] for i in self.indexes]

class CustomImageDataGenerator(Sequence):
    def __init__(self, image_paths, labels, batch_size=64, image_size=(128, 128), augment=False, **kwargs):
        super().__init__(**kwargs)  # Ensure proper handling of additional arguments
        self.image_paths = image_paths
        self.labels = labels
        self.batch_size = batch_size
        self.image_size = image_size
        self.augment = augment
        self.datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2
        ) if augment else ImageDataGenerator()
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
                # Log unreadable images
                print(f"Warning: Image at path {img_path} could not be read.")
        
        if len(images) != self.batch_size:
            # Log batch size mismatches
            print(f"Warning: Batch size mismatch. Expected {self.batch_size}, got {len(images)}. Skipping batch.")
            return self.__getitem__((index + 1) % self.__len__())
        
        images = np.array(images) / 255.0
        images, valid_labels = np.array(images), np.array(valid_labels)
        if self.augment:
            return next(self.datagen.flow(images, valid_labels, batch_size=self.batch_size))
        return images, valid_labels
    
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.image_paths))
        np.random.shuffle(self.indexes)
        self.image_paths = [self.image_paths[i] for i in self.indexes]
        self.labels = [self.labels[i] for i in self.indexes]

# Data generators
train_batch_size = 64  # You can adjust this batch size as needed
test_batch_size = 64   # You can adjust this batch size as needed

train_generator = BalancedBatchGenerator(X_train_paths, y_train, batch_size=train_batch_size, augment=True, augment_count=5)
test_generator = CustomImageDataGenerator(X_test_paths, y_test, batch_size=test_batch_size)

# Model architecture
model = Sequential([
    Input(shape=(128, 128, 3)),  # Use an Input layer
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),  # Additional Conv2D layer
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(256, activation='relu', kernel_regularizer=l2(0.01)),  # Increased units and L2 regularization
    Dropout(0.5),  # Dropout for regularization
    Dense(128, activation='relu', kernel_regularizer=l2(0.01)),  # Additional Dense layer with L2 regularization
    Dropout(0.5),  # Additional Dropout
    Dense(1, activation='sigmoid', kernel_regularizer=l2(0.01))  # L2 regularization
])

# Compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Callbacks for early stopping and saving the best model
callbacks = [
    EarlyStopping(patience=3, restore_best_weights=True),
    ModelCheckpoint('best_model.keras', save_best_only=True)  # Changed to new format
]

# Train model
history = model.fit(train_generator, epochs=10, validation_data=test_generator, callbacks=callbacks)

# Evaluate model
loss, accuracy = model.evaluate(test_generator)
print(f'Test Accuracy: {accuracy * 100:.2f}%')

# Generate predictions
y_pred = model.predict(test_generator)
y_pred = (y_pred > 0.5).astype("int32").flatten()  # Convert predictions to class labels and flatten

# Flatten y_test to match the shape of y_pred
y_test_flat = np.concatenate([batch[1] for batch in test_generator])

# Print classification report
print(classification_report(y_test_flat, y_pred))

# Save final model in the new Keras format
model.save('ai_image_recognition_model.keras')