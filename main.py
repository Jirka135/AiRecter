import os
import time
import threading
from preprocess import preprocess_images
from train import train_ai
from image_generation import call_txt2img_api, read_prompts_from_file, update_eta

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
    ai_generated_dir = '\\\\192.168.0.125\\Hvezda\\AIrect\\Fake'
    real_images_dir = '\\\\192.168.0.125\\Hvezda\\AIrect\\Real'
    train_dir = '\\\\192.168.0.125\\Hvezda\\AIrect\\Train'
    test_dir = '\\\\192.168.0.125\\Hvezda\\AIrect\\Test'

    # Preprocess images without training
    preprocess_images(ai_generated_dir, real_images_dir, train_dir, test_dir, test_size=0.001, rotate=True)
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