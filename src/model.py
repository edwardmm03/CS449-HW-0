import tensorflow as tf
import numpy as np
from get_file import get_file
from typing import Union


class Model:
    model: Union[tf.keras.models.Sequential, None] = None

    def fit(
        self,
        dataset: tf.data.Dataset,
        batch_size: np.int32,
        epochs: np.int32,
    ) -> None:
        if self.model is None:
            return
        self.model.fit(dataset.batch(batch_size), epochs=epochs)

    def predict(
        self, data: np.ndarray[np.ndarray[np.int32]]
    ) -> np.ndarray[np.int32]:
        if self.model is None:
            return
        if self.model is None:
            return
        result = self.model.predict(data)
        output: np.ndarray[np.float32] = np.zeros(len(result), dtype=np.int32)
        for i in range(len(result)):
            output[i] = result[i].argmax()
        return output

    def new(self, input_shape: np.int32, output_shape: np.int32) -> None:
        self.model = tf.keras.models.Sequential(
            [
                tf.keras.layers.Flatten(input_shape=(input_shape,)),
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(output_shape, activation="sigmoid"),
            ]
        )
        self.model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

    def save(self, file: str) -> None:
        if self.model is None:
            return
        self.model.save(get_file(file))

    def load(self, file: str) -> None:
        self.model = tf.keras.models.load_model(get_file(file))
