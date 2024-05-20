import tensorflow as tf

# Print GPU devices available
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    print(f"GPUs: {gpus}")
else:
    print("No GPUs available")