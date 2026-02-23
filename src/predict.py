import cv2
import numpy as np
import sys
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt


def preprocess_image(image_path):
    import cv2
    import numpy as np

    # 1️⃣ Load image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Image not found. Check the path.")

    # 2️⃣ Slight blur (reduce noise)
    img = cv2.GaussianBlur(img, (5, 5), 0)

    # 3️⃣ Otsu threshold + invert (digit becomes white)
    _, img = cv2.threshold(
        img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # 4️⃣ Find contours
    contours, _ = cv2.findContours(
        img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        raise ValueError("No digit found in image.")

    # Get largest contour (assume it's the digit)
    contour = max(contours, key=cv2.contourArea)

    x, y, w, h = cv2.boundingRect(contour)
    img = img[y:y+h, x:x+w]

    # Resize while keeping aspect ratio
    h, w = img.shape

    if h > w:
        new_h = 20
        new_w = int(w * (20 / h))
    else:
        new_w = 20
        new_h = int(h * (20 / w))

    img = cv2.resize(img, (new_w, new_h))

    # Create blank 28x28 and center digit
    padded = np.zeros((28, 28))
    y_offset = (28 - new_h) // 2
    x_offset = (28 - new_w) // 2
    padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = img

    # Normalize
    padded = padded / 255.0
    padded = padded.reshape(1, 28, 28, 1)

    return padded

def predict(image_path):
    model = load_model("models/mnist_cnn_model.keras")

    processed_img = preprocess_image(image_path)

    prediction = model.predict(processed_img)
    predicted_digit = np.argmax(prediction)

    print("Probabilities:", prediction[0])
    print("Predicted Digit:", predicted_digit)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python src/predict.py image_name.png")
    else:
        image_name = sys.argv[1]
        image_path = f"data/{image_name}"
        predict(image_path)
