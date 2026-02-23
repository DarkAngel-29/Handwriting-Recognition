from model import build_model
from data_loader import load_data

def train():
    X_train, y_train, X_test, y_test = load_data()

    model = build_model()

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.fit(
        X_train,
        y_train,
        epochs=5,
        validation_data=(X_test, y_test)
    )

    model.save("../models/mnist_cnn_model.keras")

if __name__ == "__main__":
    train()