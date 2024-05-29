import os
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import Callback
from keras.applications import ResNet50
from keras.regularizers import l2
from preprocess import load_preprocessed_data, ImageGenerator


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

        # Evaluate model on validation data
        val_loss, val_accuracy = self.model.evaluate(self.validation_data, verbose=0)

        # Check if this is the best model so far
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            best_model_filename = os.path.join(self.model_save_dir, 'best_model.h5')
            self.model.save(best_model_filename)

        # Log epoch, accuracy, and model filename
        with open(self.log_file, 'a') as f:
            f.write(f"Epoch: {epoch + 1}, Accuracy: {val_accuracy * 100:.2f}%, Model: {model_filename}\n")


def train_ai():
    train_dir = 'C:\\Users\\Jirka\\VScode\\AirRect\\AiRecter\\Images\\Train'
    test_dir = 'C:\\Users\\Jirka\\VScode\\AirRect\\AiRecter\\Images\\Test'
    model_save_dir = 'C:\\Users\\Jirka\\VScode\\AirRect\\AiRecter\\Models'
    log_file = 'C:\\Users\\Jirka\\VScode\\AirRect\\AiRecter\\training_log.txt'
    
    os.makedirs(model_save_dir, exist_ok=True)

    # Load preprocessed data
    X_train_paths, y_train, X_test_paths, y_test = load_preprocessed_data(train_dir, test_dir)

    train_generator = ImageGenerator(X_train_paths, y_train, batch_size=8)
    test_generator = ImageGenerator(X_test_paths, y_test, batch_size=8)

    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(512, 512, 3))
    x = base_model.output
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = BatchNormalization()(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    predictions = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

    checkpoint_and_log_callback = ModelCheckpointAndLog(model_save_dir, log_file, test_generator)

    model.fit(train_generator, epochs=25, validation_data=test_generator, callbacks=[checkpoint_and_log_callback])

    for layer in base_model.layers:
        layer.trainable = True

    model.compile(optimizer=Adam(learning_rate=0.00001), loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(train_generator, epochs=25, validation_data=test_generator, callbacks=[checkpoint_and_log_callback])

    loss, accuracy = model.evaluate(test_generator)
    print(f'Test Accuracy: {accuracy * 100:.2f}%')

    model.save(os.path.join(model_save_dir, 'ai_image_recognition_model_final.h5'))
