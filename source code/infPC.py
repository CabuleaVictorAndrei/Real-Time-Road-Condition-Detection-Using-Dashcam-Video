import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import models, transforms


def load_efficientnet_b3_model(num_classes, model_path):
    model = models.efficientnet_b3(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3, inplace=True),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, num_classes)
    )
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    model.to(device)
    model.eval()
    return model


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    num_classes = 3
    model_path = "best_model.pth"
    model = load_efficientnet_b3_model(num_classes, model_path)
    classes = ["clear", "snowy", "wet"]

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    video_path = "test/test_video.mp4"
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    cv2.namedWindow("Road Condition Classification - EfficientNet-B3", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Road Condition Classification - EfficientNet-B3", 1280, 720)

    frame_counter = 0
    process_every_n_frames = 1

    frame_counts = {cls: 0 for cls in classes}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_counter += 1
        if frame_counter % process_every_n_frames != 0:
            continue

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (128, 128))
        img_normalized = img_resized.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_normalized = (img_normalized - mean) / std
        img_chw = np.transpose(img_normalized, (2, 0, 1))
        img_tensor = torch.tensor(img_chw, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img_tensor)
            probs = F.softmax(output, dim=1)[0].cpu().numpy()

        predicted_class_idx = np.argmax(probs)
        predicted_class = classes[predicted_class_idx]
        confidence = probs[predicted_class_idx]

        frame_counts[predicted_class] += 1

        y0, dy = 30, 40
        for i, (cls, prob) in enumerate(zip(classes, probs)):
            color = (0, 255, 0) if prob == max(probs) else (255, 255, 255)
            thickness = 2 if prob == max(probs) else 1
            cv2.putText(frame, f"{cls}: {prob:.4f}", (20, y0 + i * dy),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, thickness, cv2.LINE_AA)

        prediction_text = f"Prediction: {predicted_class} ({confidence:.4f})"
        cv2.putText(frame, prediction_text, (20, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3, cv2.LINE_AA)

        resolution_text = f"Model: EfficientNet-B3 (128x128)"
        cv2.putText(frame, resolution_text, (frame.shape[1] - 400, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        frame_text = f"Frame: {frame_counter}"
        cv2.putText(frame, frame_text, (frame.shape[1] - 200, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("Road Condition Classification - EfficientNet-B3", frame)

        print(f"Frame {frame_counter} counts: {frame_counts}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Video inference completed.")
    print(f"Final counts: {frame_counts}")
