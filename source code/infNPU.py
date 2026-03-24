import cv2
import numpy as np
import time
import threading
import queue
import tflite_runtime.interpreter as tflite
from tflite_runtime.interpreter import load_delegate


video_path = "test_video.mp4"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

model_path = "model_int8_128.tflite"
npu_delegate = load_delegate("libvx_delegate.so")
interpreter = tflite.Interpreter(model_path=model_path, experimental_delegates=[npu_delegate])
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']
input_dtype = input_details[0]['dtype']

input_scale, input_zero_point = input_details[0]['quantization']
mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

classes = ["clear", "snowy", "wet"]

frame_queue = queue.Queue(maxsize=5)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def video_reader():
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_queue.put(frame)
    frame_queue.put(None)

def npu_worker():
    frame_counter = 0
    total_inference_time = 0.0
    total_loop_time = 0.0

    class_counts = {cls: 0 for cls in classes}

    while True:
        frame = frame_queue.get()
        if frame is None:
            break

        frame_counter += 1
        loop_start = time.time()

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (128, 128))
        img_normalized = (img_resized / 255.0 - mean) / std
        img_chw = np.transpose(img_normalized, (2, 0, 1))[np.newaxis, ...]
        img_int8 = (img_chw / input_scale + input_zero_point).astype(input_dtype)

        start_inf = time.time()
        interpreter.set_tensor(input_details[0]['index'], img_int8)
        interpreter.invoke()
        end_inf = time.time()

        inference_time = end_inf - start_inf
        total_inference_time += inference_time

        output_data = interpreter.get_tensor(output_details[0]['index'])
        output_scale, output_zero_point = output_details[0]['quantization']
        output_data = output_scale * (output_data.astype(np.float32) - output_zero_point)
        output_data = output_data.squeeze()

        probs = softmax(output_data)
        predicted_class_idx = np.argmax(probs)
        predicted_class = classes[predicted_class_idx]
        confidence = probs[predicted_class_idx]

        class_counts[predicted_class] += 1

        loop_end = time.time()
        loop_time = loop_end - loop_start
        total_loop_time += loop_time

        real_fps = 1.0 / loop_time if loop_time > 0 else 0
        npu_fps = 1.0 / inference_time if inference_time > 0 else 0

        print(f"Frame {frame_counter}: Pred={predicted_class}, Conf={confidence:.4f}, "
              f"Counts => Clear: {class_counts['clear']}, Snowy: {class_counts['snowy']}, Wet: {class_counts['wet']}"
              f"Inference={inference_time * 1000:.2f} ms ({npu_fps:.2f} FPS), "
              f"Real FPS={real_fps:.2f}, "
              )

    avg_npu_fps = frame_counter / total_inference_time
    avg_real_fps = frame_counter / total_loop_time
    print(f"\nProcessed {frame_counter} frames.")
    print(f"Average NPU FPS (inference only): {avg_npu_fps:.2f}")
    print(f"Average REAL FPS (full pipeline): {avg_real_fps:.2f}")


reader_thread = threading.Thread(target=video_reader)
npu_thread = threading.Thread(target=npu_worker)

start_time = time.time()
reader_thread.start()
npu_thread.start()

reader_thread.join()
npu_thread.join()
cap.release()
