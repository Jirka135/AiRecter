import os
from keras.models import Model
from keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import Callback, LearningRateScheduler, ModelCheckpoint, TensorBoard
from keras.applications import ResNet50
from keras.regularizers import l2
from preprocess import load_preprocessed_data, ImageGenerator

class ModelCheckpointAndLog(Callback):
    def __init__(self, model_save_dir, log_file, validation_data, sse, **kwargs):
        super().__init__(**kwargs)
        self.model_save_dir = model_save_dir
        self.log_file = log_file
        self.validation_data = validation_data
        self.sse = sse
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

        # Send SSE update
        self.sse.publish({"progress": (epoch + 1) / 50 * 100, "message": f"Training epoch {epoch + 1} of 50"}, type='train')

def train_ai(sse):
    train_dir = 'D:\\AIimages\\Train'
    test_dir = 'D:\\AIimages\\Test'
    model_save_dir = 'C:\\Users\\Jirka\\VScode\\AirRect\\AiRecter\\Models'
    log_file = 'C:\\Users\\Jirka\\VScode\\AirRect\\AiRecter\\training_log.txt'
    log_dir = 'logs'
    
    os.makedirs(model_save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Load preprocessed data into memory
    X_train, y_train, X_test, y_test = load_preprocessed_data(train_dir, test_dir)

    train_generator = ImageGenerator(X_train, y_train, batch_size=8)
    test_generator = ImageGenerator(X_test, y_test, batch_size=8)

    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(512, 512, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
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

    # Learning rate scheduler
    def scheduler(epoch, lr):
        if epoch < 10:
            return lr
        else:
            return lr * 0.1

    lr_scheduler = LearningRateScheduler(scheduler)
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
    checkpoint_callback = ModelCheckpoint(filepath=os.path.join(model_save_dir, 'model_epoch_{epoch:02d}.h5'), save_best_only=True)
    checkpoint_and_log_callback = ModelCheckpointAndLog(model_save_dir, log_file, test_generator, sse)

    model.fit(
        train_generator,
        epochs=25,
        validation_data=test_generator,
        callbacks=[checkpoint_callback, lr_scheduler, tensorboard_callback, checkpoint_and_log_callback]
    )

    for layer in base_model.layers:
        layer.trainable = True

    model.compile(optimizer=Adam(learning_rate=0.00001), loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(
        train_generator,
        epochs=25,
        validation_data=test_generator,
        callbacks=[checkpoint_callback, lr_scheduler, tensorboard_callback, checkpoint_and_log_callback]
    )

    loss, accuracy = model.evaluate(test_generator)
    print(f'Test Accuracy: {accuracy * 100:.2f}%')

    model.save(os.path.join(model_save_dir, 'ai_image_recognition_model_final.h5'))
    sse.publish({"progress": 100, "message": "Training completed!"}, type='train')
