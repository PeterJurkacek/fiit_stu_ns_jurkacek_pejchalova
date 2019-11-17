import sys

sys.path.append('/labs')
print(sys.path)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D

def get_model(image_height, image_width):
    return Sequential([
        Conv2D(16, 3, padding='same', activation='relu', input_shape=(image_height, image_width, 3)),
        MaxPooling2D(),
        Conv2D(32, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Conv2D(64, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Flatten(),
        Dense(512, activation='relu'),
        # Dropout(0.25),
        Dense(1, activation='sigmoid')
    ])
