import numpy as np
import matplotlib.pylab as plt
from PIL import Image
from keras.models import load_model,model_from_json
from skimage.transform import rescale, resize, downscale_local_mean
import cv2
import time
# Returns a compiled model identical to the previous one
# model = load_model('model.h5')
# dimensions of our images.
img_width, img_height = 224, 224

batch_size = 1

# load json and create model
json_file = open("new_mobile_model_98_94_18_light.json", 'r')
loaded_model_json = json_file.read()
json_file.close()
#with CustomObjectScope({'relu6': applications.mobilenet.relu6,'DepthwiseConv2D': applications.mobilenet.DepthwiseConv2D}):
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("new_mobile_model_98_94_18_light.h5")
print("Loaded model from disk")
# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#score = loaded_model.evaluate(X, Y, verbose=0)

img = cv2.imread("./examples/result_h0999.jpeg")
# print(np.array(img))
img = cv2.resize(img,(img_width,img_height))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = np.reshape(img,[1,img_height,img_width,3])
img = img / 255.0
img = img - np.array([103.939/255, 116.779/255, 123.68/255],dtype=np.float32).reshape(1,1,1,3)
start = time.time()
preds = loaded_model.predict(img, batch_size=batch_size)
end = time.time()
# print("\n### prediction time: " + str(end - start))
print("## prediction: result_h0999.jpeg", preds*100)

# img = Image.open("./examples/result_h0999.jpeg")
# img = np.asarray(img)
# print(np.array(img))
# # print("\n### Image loaded ",img.shape)
# img = resize(img, (224,224),anti_aliasing=False)
# img = np.array(img, dtype=np.float32) / 255.0
# img = np.reshape(img,[1,224,224,3])
# img = img - np.array([103.939/255, 116.779/255, 123.68/255],dtype=np.float32).reshape(1,1,1,3)
# #print(img[0,:3,:,:])
# # start = time.time()
# preds = loaded_model.predict(img, batch_size=batch_size)
# # end = time.time()
# print("## prediction: result_h0999.jpeg", preds*100)
# print("## prediction time: " + str(end - start))

img = cv2.imread("./examples/result_h1000.jpeg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img,(img_width,img_height))
img = np.reshape(img,[1,img_height,img_width,3])
img = img / 255.0
img = img - np.array([103.939/255, 116.779/255, 123.68/255],dtype=np.float32).reshape(1,1,1,3)
start = time.time()
preds = loaded_model.predict(img, batch_size=batch_size)
end = time.time()
# print("\n### prediction time: " + str(end - start))
print("## prediction: result_h1000.jpeg", preds*100)

img = cv2.imread("./examples/result_t0778.jpeg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img,(img_width,img_height))
img = np.reshape(img,[1,img_height,img_width,3])
img = img / 255.0
img = img - np.array([103.939/255, 116.779/255, 123.68/255],dtype=np.float32).reshape(1,1,1,3)
start = time.time()
preds = loaded_model.predict(img, batch_size=batch_size)
end = time.time()
# print("\n### prediction time: " + str(end - start))
print("## prediction: result_t0778.jpeg", preds*100)

img = cv2.imread("./examples/result_t0997.jpeg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img,(img_width,img_height))
img = np.reshape(img,[1,img_height,img_width,3])
img = img / 255.0
img = img - np.array([103.939/255, 116.779/255, 123.68/255],dtype=np.float32).reshape(1,1,1,3)
start = time.time()
preds = loaded_model.predict(img, batch_size=batch_size)
end = time.time()
# print("\n### prediction time: " + str(end - start))
print("## prediction: result_t0997.jpeg", preds*100)

# # img = plt.imread("./m.png")
# # load image with Pillow as RGB
# img = Image.open("./examples/result_h0999.jpeg")
# img = np.asarray(img)
# # print("\n### Image loaded ",img.shape)
# img = resize(img, (224,224),anti_aliasing=False)
# # print("\n### Image loaded ",img.shape)
# # img = np.images.resize(img,(img_width,img_height))
# img = np.array(img, dtype=np.float32) / 255.0
# img = np.reshape(img,[1,224,224,3])
# img = img - np.array([103.939/255, 116.779/255, 123.68/255],dtype=np.float32).reshape(1,1,1,3)
# #print(img[0,:3,:,:])
# # start = time.time()
# preds = model.predict(img)
# # end = time.time()
# print("## prediction: result_h0999.jpeg", preds*100)
# # print("## prediction time: " + str(end - start))
# ########################################################
# img = Image.open("./examples/result_h1000.jpeg")
# img = np.asarray(img)
# img = resize(img, (224,224),anti_aliasing=False)
# # img = np.images.resize(img,(img_width,img_height))
# img = np.array(img, dtype=np.float32) / 255.0
# img = np.reshape(img,[1,224,224,3])
# img = img - np.array([103.939/255, 116.779/255, 123.68/255],dtype=np.float32).reshape(1,1,1,3)
# # start = time.time()
# preds = model.predict(img)
# # end = time.time()
# print("## prediction:result_h1000.jpeg ", preds*100)
# ########################################################
# img = Image.open("./examples/result_t0778.jpeg")
# img = np.asarray(img)
# img = resize(img, (224,224),anti_aliasing=False)
# # img = np.images.resize(img,(img_width,img_height))
# img = np.array(img, dtype=np.float32) / 255.0
# img = np.reshape(img,[1,224,224,3])
# img = img - np.array([103.939/255, 116.779/255, 123.68/255],dtype=np.float32).reshape(1,1,1,3)
# # start = time.time()
# preds = model.predict(img)
# # end = time.time()
# print("## prediction:result_t0778.jpeg ", preds*100)
# ########################################################
# img = Image.open("./examples/result_t0997.jpeg")
# img = np.asarray(img)
# img = resize(img, (224,224),anti_aliasing=False)
# # img = np.images.resize(img,(img_width,img_height))
# img = np.array(img, dtype=np.float32) / 255.0
# img = np.reshape(img,[1,224,224,3])
# img = img - np.array([103.939/255, 116.779/255, 123.68/255],dtype=np.float32).reshape(1,1,1,3)
# # start = time.time()
# preds = model.predict(img)
# # end = time.time()
# print("## prediction:result_t0997.jpeg ", preds*100)