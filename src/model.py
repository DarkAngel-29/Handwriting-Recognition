from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input

def build_model():
    model = Sequential([
        Input(shape=(28,28,1)),

        Conv2D(32, (3,3), activation='relu'),
        MaxPooling2D((2,2)),

        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D((2,2)),

        Flatten(),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ])

    return model