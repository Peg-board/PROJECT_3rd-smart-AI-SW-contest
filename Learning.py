import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models

class ML:

    # cv2.imwrite('./savedImages/channel_%d(%d, %d, %d).png' % (i, int(header[0]), int(header[1]), int(header[2])), img)
    def __init__(self,train_images, train_labels, width, height):
        #np.reshape(이미지 배열, (배열 길이, 이미지 넓이 크기, 이미지 높이 크기, 차원) )
        self.train_images = np.reshape(train_images, (len(train_images), width, height, 1))
        self.train_labels = train_labels
        self.width = width
        self.height = height

    def create(self):
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(self.width, self.height, 1)),
            tf.keras.layers.MaxPooling2D((2,2)),
            tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),

            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            #Dense(분별할 종류 개수, activation='aaa')
            tf.keras.layers.Dense(4, activation='softmax'),

            #tf.keras.layers.Dropout(0.2),
            # tf.keras.layers.Dense(4, activation='softmax')
        ])
        self.model.summary()

    def train(self):
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        self.model.fit(self.train_images, self.train_labels, epochs=30)

    def test(self, input):
        # 모델을 불러와서 input이미지로
        return
