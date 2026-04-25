import tensorflow as tf
from tensorflow.keras import layers, models

train_ds = tf.keras.utils.image_dataset_from_directory(
    "data/chest_xray/train",
    image_size=(224, 224),
    batch_size=32
)

model = models.Sequential([
    layers.Rescaling(1./255, input_shape=(224,224,3)),

    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(2, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(train_ds, epochs=5)

model.save("models/cnn_model.h5")
print("CNN model saved.")