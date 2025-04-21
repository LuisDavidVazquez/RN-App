import tensorflow as tf
import os

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Define input and output paths
model_path = os.path.join(current_dir, 'mejor_modelo_ft.h5')
output_path = os.path.join(current_dir, 'mejor_modelo_ft.tflite')

# Load the Keras model
print(f"Loading model from: {model_path}")
model = tf.keras.models.load_model(model_path)

# Convert the model
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Configure the converter for better compatibility
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,  # Use only TFLite built-in ops
    tf.lite.OpsSet.SELECT_TF_OPS  # Include regular TensorFlow ops if needed
]
converter.target_spec.supported_types = [tf.float32]
converter.allow_custom_ops = False

# Convert the model
print("Converting model to TFLite format...")
tflite_model = converter.convert()

# Save the model
print(f"Saving TFLite model to: {output_path}")
with open(output_path, 'wb') as f:
    f.write(tflite_model)

print("Conversion completed successfully!")
