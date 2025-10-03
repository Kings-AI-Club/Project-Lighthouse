import tensorflow as tf
from tensorflow import keras

class ThreeClassFNN(keras.Model):
    def __init__(self):
        super().__init__()
        self.dropout = keras.layers.Dropout(0.3)
        self.dense1 = keras.layers.Dense(32, activation='relu')
        self.dense2 = keras.layers.Dense(32, activation='relu')
        self.dense3 = keras.layers.Dense(32, activation='relu')
        self.dense4 = keras.layers.Dense(32, activation='relu')
        self.out = keras.layers.Dense(3, activation='softmax')
    
    def call(self, inputs, training=False):
        output = self.dense1(inputs)
        output = self.dropout(output)
        output = self.dense2(output)
        output = self.dropout(output)
        output = self.dense3(output)
        output = self.dropout(output)
        output = self.dense4(output)
        return self.out(output)

    def build(self, input_shape):
        super().build(input_shape)