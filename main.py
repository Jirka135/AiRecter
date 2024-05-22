import os
import cv2
import numpy as np
from preprocess import preprocess_images, load_preprocessed_data
from image_generation import call_txt2img_api
from image_generation import read_prompts_from_file
from image_generation import update_eta
from keras._tf_keras.keras.utils import Sequence
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import threading
import time

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

if __name__ == '__main__':
    ai_generated_dir = 'C:\\Users\\Jirka\\VScode\\AirRect\\AiRecter\\Images\\Fake'
    real_images_dir = 'C:\\Users\\Jirka\\VScode\\AirRect\\AiRecter\\Images\\Real'
    train_dir = 'C:\\Users\\Jirka\\VScode\\AirRect\\AiRecter\\Images\\Train'
    test_dir = 'C:\\Users\\Jirka\\VScode\\AirRect\\AiRecter\\Images\\Test'
    prompt_file = 'prompts.txt'

    preprocess_images(ai_generated_dir, real_images_dir, train_dir, test_dir, test_size=0.2)

    X_train_paths, y_train, X_test_paths, y_test = load_preprocessed_data(train_dir, test_dir)

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

    prompts = read_prompts_from_file(prompt_file)
    if not prompts:
        print("No prompts found in the file. Please add prompts to 'prompts.txt' and run the script again.")
    else:
        total_prompts = len(prompts)
        start_time = time.time()
        processed_prompts = [0]
        stop_event = threading.Event()

        eta_thread = threading.Thread(target=update_eta, args=(total_prompts, start_time, processed_prompts, stop_event))
        eta_thread.start()

        try:
            for prompt in prompts:
                call_txt2img_api(prompt)
                processed_prompts[0] += 1  # Increment the count of processed prompts
        finally:
            stop_event.set()
            eta_thread.join()
        print("\nAll prompts processed.")
