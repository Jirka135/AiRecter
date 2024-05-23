import os
import time
import threading
from preprocess import preprocess_images, load_preprocessed_data, ImageDataGenerator
from image_generation import call_txt2img_api, read_prompts_from_file, update_eta
from keras.models import Sequential, Model
from keras.layers import Input, Add, Dense, Activation, BatchNormalization, Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dropout,concatenate
from keras.optimizers import Adam
from keras.callbacks import Callback



class ModelCheckpointAndLog(Callback):
    def __init__(self, model_save_dir, log_file, validation_data):
        super().__init__()
        self.model_save_dir = model_save_dir
        self.log_file = log_file
        self.validation_data = validation_data

    def on_epoch_end(self, epoch, logs=None):
        # Save model
        model_filename = os.path.join(self.model_save_dir, f'model_epoch_{epoch + 1}.h5')
        self.model.save(model_filename)
        
        # Evaluate model on validation data
        val_loss, val_accuracy = self.model.evaluate(self.validation_data, verbose=0)
        
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

    # Define and compile the model
    model = Sequential([
        Conv2D(64, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Conv2D(256, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Conv2D(512, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

    # Create data generators
    train_generator = ImageDataGenerator(X_train_paths, y_train, batch_size=1)
    test_generator = ImageDataGenerator(X_test_paths, y_test, batch_size=1)

    # Callback to save model and log after each epoch
    checkpoint_and_log_callback = ModelCheckpointAndLog(model_save_dir, log_file, test_generator)

    # Train model
    model.fit(train_generator, epochs=1, validation_data=test_generator, callbacks=[checkpoint_and_log_callback])

    # Evaluate model
    loss, accuracy = model.evaluate(test_generator)
    print(f'Test Accuracy: {accuracy * 100:.2f}%')

    # Save final model
    model.save('ai_image_recognition_model.h5')

def generate_images():
    prompt_file = 'prompts.txt'
    prompts = read_prompts_from_file(prompt_file)
    if not prompts:
        print("No prompts found in the file. Please add prompts to 'prompts.txt' and run the script again.")
    else:
        negative_prompt = "low quality, blurry, bad anatomy, disfigured, deformed, extra limbs, extra fingers, extra arms, extra legs, poorly drawn, poorly rendered, bad proportions, unnatural body, unnatural lighting, watermark, text, logo, nsfw, low resolution, grainy, overexposed, underexposed, distorted, cropped, jpeg artifacts, watermark, cartoon, sketch, 3d render, anime, unrealistic, bad art, ugly, messy, cluttered, low detail, noise, overprocessed"
        total_prompts = len(prompts)
        start_time = time.time()
        processed_prompts = [0]
        stop_event = threading.Event()

        # Start the ETA update thread
        eta_thread = threading.Thread(target=update_eta, args=(total_prompts, start_time, processed_prompts, stop_event))
        eta_thread.start()

        try:
            for prompt in prompts:
                call_txt2img_api(prompt, negative_prompt)
                processed_prompts[0] += 1  # Increment the count of processed prompts
        finally:
            stop_event.set()
            eta_thread.join()
        print("\nAll prompts processed.")

def preprocess_only():
    ai_generated_dir = 'C:\\Users\\Jirka\\VScode\\AirRect\\AiRecter\\Images\\Fake'
    real_images_dir = 'C:\\Users\\Jirka\\VScode\\AirRect\\AiRecter\\Images\\Real'
    train_dir = 'C:\\Users\\Jirka\\VScode\\AirRect\\AiRecter\\Images\\Train'
    test_dir = 'C:\\Users\\Jirka\\VScode\\AirRect\\AiRecter\\Images\\Test'

    # Preprocess images without training
    preprocess_images(ai_generated_dir, real_images_dir, train_dir, test_dir, test_size=0.2)
    print("Preprocessing completed.")

def main():
    while True:
        print("Menu:")
        print("1. Train AI")
        print("2. Generate Images")
        print("3. Preprocess Images")
        print("4. Exit")
        choice = input("Enter your choice: ")

        if choice == '1':
            train_ai()
        elif choice == '2':
            generate_images()
        elif choice == '3':
            preprocess_only()
        elif choice == '4':
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == '__main__':
    main()
