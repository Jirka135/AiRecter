import os
import logging
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Conv2D, GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import Callback, LearningRateScheduler, EarlyStopping
from keras.applications import ResNet50
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from preprocess import load_preprocessed_data
from sklearn.utils import shuffle

# Set up logging configuration
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class ModelCheckpointAndLog(Callback):
    def __init__(self, model_save_dir, log_file, validation_data, **kwargs):
        super().__init__(**kwargs)
        self.model_save_dir = model_save_dir
        self.log_file = log_file
        self.validation_data = validation_data
        self.best_val_loss = float('inf')

    def on_epoch_end(self, epoch, logs=None):
        # Save model
        model_filename = os.path.join(self.model_save_dir, f'model_epoch_{epoch + 1}.h5')
        self.model.save(model_filename)
        logging.info(f"Model saved: {model_filename}")

        # Evaluate model on validation data
        val_loss, val_accuracy = self.model.evaluate(self.validation_data, verbose=0)
        logging.info(f"Validation loss: {val_loss:.10f}, Validation accuracy: {val_accuracy:.10f}")

        # Check if this is the best model so far
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            best_model_filename = os.path.join(self.model_save_dir, 'best_model.h5')
            self.model.save(best_model_filename)
            logging.info(f"Best model saved: {best_model_filename}")

        # Log epoch, accuracy, and model filename
        log_message = f"Epoch: {epoch + 1}, Accuracy: {val_accuracy * 100:.10f}%, Model: {model_filename}"
        with open(self.log_file, 'a') as f:
            f.write(log_message + "\n")
        logging.info(log_message)

def build_simplified_model():
    logging.info("Building simplified model...")
    
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(512, 512, 3))
    for layer in base_model.layers[:-4]:
        layer.trainable = False

    x = base_model.output
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(1024, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(2048, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    predictions = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    logging.info("Simplified model built successfully with an additional convolutional layer")
    return model

def lr_schedule(epoch):
    initial_lr = 1e-3
    drop = 0.5
    epochs_drop = 5
    lr = initial_lr * (drop ** (epoch // epochs_drop))
    return lr

def load_data_into_memory(paths, labels, batch_size, augment=False):
    def preprocess_image(path, label):
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, [512, 512])
        image = image / 255.0
        return image, label

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    dataset = dataset.map(preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    if augment:
        datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True)
        dataset = dataset.map(lambda x, y: (tf.py_function(func=datagen.random_transform, inp=[x], Tout=tf.float32), y), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset

def compile_and_train(model, train_data, validation_data, model_save_dir, log_file, epochs=50):
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    
    lr_scheduler = LearningRateScheduler(lr_schedule)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    checkpoint_and_log_callback = ModelCheckpointAndLog(model_save_dir, log_file, validation_data)
    
    history = model.fit(
        train_data,
        epochs=epochs,
        validation_data=validation_data,
        callbacks=[lr_scheduler, checkpoint_and_log_callback, early_stopping]
    )
    
    return history

def plot_training_history(history, save_dir):
    # Plot training & validation accuracy values
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_history.png'))
    plt.show()

def train_ai():
    logging.info("Starting AI training...")
    train_dir = 'C:\\valAI\\Train'
    test_dir = 'C:\\valAI\\Test'
    model_save_dir = 'C:\\Users\\Jirka\\VScode\\AirRect\\AiRecter\\Models'
    log_file = 'C:\\Users\\Jirka\\VScode\\AirRect\\AiRecter\\training_log.txt'
    
    os.makedirs(model_save_dir, exist_ok=True)

    # Load preprocessed data
    logging.info("Loading preprocessed data...")
    X_train_paths, y_train, X_test_paths, y_test = load_preprocessed_data(train_dir, test_dir)
    X_train_paths, y_train = shuffle(X_train_paths, y_train, random_state=42)
    X_test_paths, y_test = shuffle(X_test_paths, y_test, random_state=42)

    train_dataset = load_data_into_memory(X_train_paths, y_train, batch_size=32, augment=True)
    test_dataset = load_data_into_memory(X_test_paths, y_test, batch_size=32)

    logging.info(f"Loaded {len(X_train_paths)} training images and {len(X_test_paths)} testing images")

    model = build_simplified_model()
    
    history = compile_and_train(model, train_dataset, test_dataset, model_save_dir, log_file, epochs=10)
    
    logging.info("Training completed")

    loss, accuracy = model.evaluate(test_dataset)
    logging.info(f'Test Accuracy: {accuracy * 100:.10f}%')

    model.save(os.path.join(model_save_dir, 'ai_image_recognition_model_final.h5'))
    logging.info("Final model saved")

    plot_training_history(history, model_save_dir)

if __name__ == '__main__':
    train_ai()