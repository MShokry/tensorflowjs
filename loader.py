# MLP for Pima Indians Dataset Serialize to JSON and HDF5
from keras.models import model_from_json
import numpy
import tensorflowjs as tfjs

# load json and create model
json_file = open('new_mobile_model_98_94_18_light.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("new_mobile_model_98_94_18_light.h5")
print("Loaded model from disk")
loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
loaded_model.save('model.h5')
tfjs.converters.save_keras_model(loaded_model, "tfjs")
