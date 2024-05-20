from datetime import datetime
import urllib.request
import base64
import json
import time
import os

webui_server_url = 'http://127.0.0.1:7860'

out_dir = 'api_out'
out_dir_t2i = os.path.join(out_dir, 'txt2img')
os.makedirs(out_dir_t2i, exist_ok=True)

def timestamp():
    return datetime.fromtimestamp(time.time()).strftime("%Y%m%d-%H%M%S")

def decode_and_save_base64(base64_str, save_path):
    with open(save_path, "wb") as file:
        file.write(base64.b64decode(base64_str))

def call_api(api_endpoint, **payload):
    data = json.dumps(payload).encode('utf-8')
    request = urllib.request.Request(
        f'{webui_server_url}/{api_endpoint}',
        headers={'Content-Type': 'application/json'},
        data=data,
    )
    response = urllib.request.urlopen(request)
    return json.loads(response.read().decode('utf-8'))

def call_txt2img_api(prompt, negative_prompt, seed=-1, steps=32, width=512, height=512, cfg_scale=18, sampler_name="Euler a", n_iter=5, batch_size=2, enable_hr=True, hr_scale=2, hr_upscaler="R-ESRGAN 4x+", denoising_strength=0.7):
    payload = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "seed": seed,
        "steps": steps,
        "width": width,
        "height": height,
        "cfg_scale": cfg_scale,
        "sampler_name": sampler_name,
        "n_iter": n_iter,
        "batch_size": batch_size,
        "enable_hr": enable_hr,
        "hr_scale": hr_scale,
        "hr_upscaler": hr_upscaler,
        "denoising_strength": denoising_strength
    }
    response = call_api('sdapi/v1/txt2img', **payload)
    for index, image in enumerate(response.get('images')):
        save_path = os.path.join(out_dir_t2i, f'txt2img-{timestamp()}-{index}.png')
        decode_and_save_base64(image, save_path)

def read_prompts_from_file(file_path):
    if not os.path.exists(file_path):
        with open(file_path, 'w') as file:
            pass  # Create an empty file
        print(f"{file_path} did not exist, so it was created. Please add your prompts and run the script again.")
        return []
    with open(file_path, 'r') as file:
        content = file.read()
    prompts = content.split(';')
    return prompts

if __name__ == '__main__':
    prompts = read_prompts_from_file('prompts.txt')
    if not prompts:
        print("No prompts found in the file. Please add prompts to 'prompts.txt' and run the script again.")
    else:
        negative_prompt = "low quality, blurry, bad anatomy, disfigured, deformed, extra limbs, extra fingers, extra arms, extra legs, poorly drawn, poorly rendered, bad proportions, unnatural body, unnatural lighting, watermark, text, logo, nsfw, low resolution, grainy, overexposed, underexposed, distorted, cropped, jpeg artifacts, watermark, cartoon, sketch, 3d render, anime, unrealistic, bad art, ugly, messy, cluttered, low detail, noise, overprocessed"
        for prompt in prompts:
            call_txt2img_api(prompt, negative_prompt)