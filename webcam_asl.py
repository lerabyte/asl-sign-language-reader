from pathlib import Path
import json
from collections import deque

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras

OUTPUT_DIR = Path("asl_model_output")
MODEL_PATH = OUTPUT_DIR / "best_asl_model.keras"
METADATA_PATH = OUTPUT_DIR / "asl_metadata.json"

CAMERA_NUMBER = 0
CONFIDENCE_THRESHOLD = 0.70
SMOOTHING_FRAMES = 8
BOX_SCALE = 0.75
MIRROR_CAMERA = True


def load_model_and_metadata():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Could not find model: {MODEL_PATH}\n"
            "Run train_asl_model.py first."
        )

    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Could not find metadata: {METADATA_PATH}\n"
            "Run train_asl_model.py first."
        )

    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    class_names = metadata["class_names"]
    img_size = tuple(metadata["image_size"])

    model = keras.models.load_model(MODEL_PATH)
    return model, class_names, img_size


def main():
    model, class_names, img_size = load_model_and_metadata()

    cap = cv2.VideoCapture(CAMERA_NUMBER)

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera {CAMERA_NUMBER}. "
            "Try changing CAMERA_NUMBER to 1."
        )

    prediction_history = deque(maxlen=SMOOTHING_FRAMES)

    print("Webcam started.")
    print("Put your hand inside the white box.")
    print("Press q to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if MIRROR_CAMERA:
            frame = cv2.flip(frame, 1)

        h, w = frame.shape[:2]

        box_size = int(min(h, w) * BOX_SCALE)
        box_size = max(50, min(box_size, min(h, w)))

        cx = w // 2
        cy = h // 2

        x1 = max(0, cx - box_size // 2)
        y1 = max(0, cy - box_size // 2)
        x2 = min(w, x1 + box_size)
        y2 = min(h, y1 + box_size)

        x1 = max(0, x2 - box_size)
        y1 = max(0, y2 - box_size)

        crop = frame[y1:y2, x1:x2]

        resized = cv2.resize(crop, img_size)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        input_batch = np.expand_dims(rgb.astype("float32"), axis=0)

        probs = model.predict(input_batch, verbose=0)[0]

        prediction_history.append(probs)
        smooth_probs = np.mean(prediction_history, axis=0)

        best_index = int(np.argmax(smooth_probs))
        best_label = class_names[best_index]
        best_confidence = float(smooth_probs[best_index])

        if best_confidence < CONFIDENCE_THRESHOLD:
            shown_label = "?"
        else:
            shown_label = best_label

        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

        main_text = f"{shown_label}  {best_confidence:.1%}"
        cv2.putText(
            frame,
            main_text,
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )

        top_indices = smooth_probs.argsort()[-3:][::-1]
        y_text = 90

        for index in top_indices:
            text = f"{class_names[int(index)]}: {float(smooth_probs[int(index)]):.1%}"
            cv2.putText(
                frame,
                text,
                (30, y_text),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            y_text += 32

        cv2.imshow("ASL Alphabet Detector", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
