# classifier.py – uses separate tf_keras alias
import tensorflow as tf
import tf_keras as tf_keras
import tensorflow_hub as hub
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split

class DogBreedClassifier:
    """MobileNetV2 dog-breed classifier with tf_keras imports separated."""

    def __init__(self,
                 labels_csv_path,
                 images_dir,
                 image_size=224,
                 batch_size=32,
                 val_split=0.2):
        self.labels_csv_path = labels_csv_path
        self.images_dir = images_dir
        self.IMAGE_SIZE = image_size
        self.BATCH_SIZE = batch_size
        self.VAL_SPLIT = val_split
        self.model = None
        self.unique_breeds = None
        self._load_labels()

    # ------------------------------------------------------------------
    def _load_labels(self):
        labels_df = pd.read_csv(self.labels_csv_path)
        image_paths = [os.path.join(self.images_dir, f"{img_id}.jpg") for img_id in labels_df['id']]
        labels = labels_df['breed'].to_numpy()
        self.unique_breeds = np.unique(labels)
        one_hot_labels = np.array([breed == self.unique_breeds for breed in labels], dtype=np.float32)

        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(
            image_paths,
            one_hot_labels,
            test_size=self.VAL_SPLIT,
            random_state=42,
            stratify=one_hot_labels
        )

    # ------------------------------------------------------------------
    def _process_image(self, image_path):
        img = tf.io.read_file(image_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize(img, [self.IMAGE_SIZE, self.IMAGE_SIZE])
        return img

    def _load_and_label(self, path, label):
        return self._process_image(path), label

    def _build_dataset(self, x, y, training=True):
        ds = tf.data.Dataset.from_tensor_slices((x, y))
        if training:
            ds = ds.shuffle(buffer_size=len(x))
        ds = ds.map(self._load_and_label, num_parallel_calls=tf.data.AUTOTUNE)
        return ds.batch(self.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    # ------------------------------------------------------------------
    def create_model(self):
        hub_url = "https://tfhub.dev/google/imagenet/mobilenet_v2_130_224/classification/5"
        base_model = hub.KerasLayer(hub_url,
                                    trainable=True,
                                    input_shape=(self.IMAGE_SIZE, self.IMAGE_SIZE, 3))

        model = tf_keras.Sequential([
            base_model,
            tf_keras.layers.Dense(len(self.unique_breeds), activation='softmax')
        ])

        model.compile(
            optimizer=tf_keras.optimizers.Adam(1e-4),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        self.model = model
        return model

    # ------------------------------------------------------------------
    def train(self, epochs=25):
        train_ds = self._build_dataset(self.x_train, self.y_train, training=True)
        val_ds = self._build_dataset(self.x_val, self.y_val, training=False)

        callbacks = [
            tf_keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True),
            tf_keras.callbacks.ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True)
        ]

        self.model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks)

    # ------------------------------------------------------------------
    def load_trained_model(self, model_path='best_model.h5'):
        self.create_model()
        self.model.load_weights(model_path)

    def predict_image(self, image_path, top_k=3):
        if self.model is None:
            raise ValueError('Model not loaded. Call load_trained_model() first.')

        img = self._process_image(image_path)
        img = tf.expand_dims(img, axis=0)
        preds = self.model.predict(img)[0]
        top_indices = preds.argsort()[-top_k:][::-1]
        return [{
            'breed': self.unique_breeds[i],
            'confidence': round(float(preds[i] * 100), 2)
        } for i in top_indices]

# End of classifier.py
