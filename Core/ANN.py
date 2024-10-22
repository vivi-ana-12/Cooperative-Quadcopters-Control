import tensorflow as tf
import pandas as pd
# import warnings

class ANN:
    def __init__(self):
        self.model = tf.keras.models.load_model(".\\models\\ANN_64-32-16-8 Fine-Tuning 2_16delays.h5") 
        # warnings.filterwarnings('ignore', category=UserWarning, message='.*decay.*')
        
    def tfliteModelConfiguration(self):
        # Convert to TensorFlow Lite
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        self.tflite_model = converter.convert()
    
    def predict(self,actualInputs):
        actualPrediction = pd.DataFrame(self.model.predict(actualInputs,verbose=0))
        return actualPrediction