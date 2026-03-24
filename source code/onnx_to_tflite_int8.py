import onnx
from onnx_tf.backend import prepare
import tensorflow as tf
import numpy as np
from PIL import Image
import os

onnx_model = onnx.load("model_128.onnx")
tf_rep = prepare(onnx_model)

saved_model_dir = "model_tflite_int8_128_2"
tf_rep.export_graph(saved_model_dir)
print(f"TensorFlow SavedModel saved to {saved_model_dir}")

dataset_dir = "representative_images"
image_size = 128

def representative_data_gen(dataset_dir, image_size):
    count = 0
    for class_name in os.listdir(dataset_dir):
        class_dir = os.path.join(dataset_dir, class_name)
        print(class_dir)
        if not os.path.isdir(class_dir):
            continue
        for img_file in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_file)
            img = Image.open(img_path).convert("RGB")
            img = img.resize((image_size, image_size))
            img = np.array(img, dtype=np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            img = (img - mean) / std
            img = np.transpose(img, (2, 0, 1))  # CHW
            img = np.expand_dims(img, axis=0)   # batch size 1
            yield [img.astype(np.float32)]
            count += 1

            if count >= 100:
                return


converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

# Set int8 input and output
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model_path = "model_int8_128_2.tflite"
tflite_model = converter.convert()
with open(tflite_model_path, "wb") as f:
    f.write(tflite_model)
