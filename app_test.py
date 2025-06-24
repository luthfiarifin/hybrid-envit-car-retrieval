import cv2
from ultralytics import YOLO
import os
import torch
from PIL import Image
import numpy as np
from models.classification.model import EfficientNetB4_CBAM


def load_local_labels(filename="cars_classes.txt"):
    """Loads class labels from a local text file."""
    if not os.path.exists(filename):
        print(f"Error: Labels file not found at '{filename}'")
        print("Please run the 'train_vit.py' script first to generate this file.")
        return None
    with open(filename, "r") as f:
        labels = [line.strip() for line in f.readlines()]
    return labels


def load_detection_model(model_path):
    try:
        detection_model = YOLO(model_path)
        name_to_index = {v: k for k, v in detection_model.names.items()}
        car_class_index = name_to_index["car"]
        return detection_model, car_class_index
    except Exception as e:
        print(f"Error loading detection model '{model_path}': {e}")
        return None, None


def load_classification_model(model_path, num_classes, device):
    try:
        model = EfficientNetB4_CBAM(num_classes=num_classes)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()

        import torchvision.transforms as T

        data_config = {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "input_size": (3, 224, 224),
        }
        transforms__ = T.Compose(
            [
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=data_config["mean"], std=data_config["std"]),
            ]
        )

        car_model_labels = load_local_labels("models/classification/class_names.txt")
        return model, transforms__, car_model_labels
    except Exception as e:
        print(f"Error loading classification model: {e}")
        return None, None, None


def process_video(
    video_path,
    detection_model,
    car_class_index,
    classification_model,
    transforms_,
    car_model_labels,
    device,
    yolo_confidence_threshold=0.4,
):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file '{video_path}'")
        return
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_delay = int(1000 / fps) if fps > 0 else 40

    print(
        f"\nStarting detection on '{os.path.basename(video_path)}'. Press 'q' to exit."
    )
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("End of video reached.")
            break

        detection_results = detection_model(
            frame,
            classes=[car_class_index],
            conf=yolo_confidence_threshold,
            verbose=False,
        )

        for result in detection_results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                try:
                    car_crop_np = frame[y1:y2, x1:x2]
                    car_crop_pil = Image.fromarray(
                        cv2.cvtColor(car_crop_np, cv2.COLOR_BGR2RGB)
                    )

                    input_tensor = transforms_(car_crop_pil).unsqueeze(0).to(device)
                    with torch.no_grad():
                        output = classification_model(input_tensor)

                    probabilities = torch.nn.functional.softmax(output[0], dim=0)
                    top1_prob, top1_catid = torch.topk(probabilities, 1)

                    car_category = car_model_labels[top1_catid[0]]
                    confidence_score = top1_prob[0].item()

                    label = f"{car_category} ({confidence_score:.2f})"

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        label,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                    )
                except Exception:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(
                        frame,
                        "Car (Error)",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2,
                    )

        cv2.imshow("Car Detection and Fine-Tuned Classification", frame)

        if cv2.waitKey(frame_delay) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Player closed.")


def main():
    VIDEO_PATH = "traffic_test.mp4"
    YOLO_CONFIDENCE_THRESHOLD = 0.4
    DETECTION_MODEL_NAME = (
        "models/detection/yolo_finetune/vehicle_detection/weights/best.pt"
    )
    FINETUNED_CLASSIFIER_PATH = "models/classification/results/car_classifier_model_20250624_014531_best_acc.pth"
    NUM_CLASSES = 8  # Indonesian car types

    if not os.path.exists(VIDEO_PATH):
        print(f"Error: Video file not found at '{VIDEO_PATH}'")
        return
    if not os.path.exists(FINETUNED_CLASSIFIER_PATH):
        print(f"Error: Fine-tuned model not found at '{FINETUNED_CLASSIFIER_PATH}'")
        print("Please run 'train_vit.py' to create the model file first.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    detection_model, car_class_index = load_detection_model(DETECTION_MODEL_NAME)
    if detection_model is None:
        return

    classification_model, transforms_, car_model_labels = load_classification_model(
        FINETUNED_CLASSIFIER_PATH, NUM_CLASSES, device
    )
    if classification_model is None or not car_model_labels:
        return

    process_video(
        VIDEO_PATH,
        detection_model,
        car_class_index,
        classification_model,
        transforms_,
        car_model_labels,
        device,
        YOLO_CONFIDENCE_THRESHOLD,
    )


if __name__ == "__main__":
    main()
