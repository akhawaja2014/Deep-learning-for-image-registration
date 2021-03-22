
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
import numpy as np
import cv2
import matplotlib.pyplot as plt

model = VGG16(weights='imagenet', include_top=False)
model =Model(inputs=model.inputs, outputs=model.layers[17].output)
model.summary()

img_path = '/home/tgiencov/Registration Codes/Python image registration/Deep-learning-for-image-registration/feature_extraction/im1.JPG'

img = image.load_img(img_path, target_size=(224, 224)) #Loads an image into PIL format.
img_data = image.img_to_array(img) #Converts a PIL Image instance to a Numpy array.
print('img to array is')
print(img_data)
print(img_data.shape)
img_data = np.expand_dims(img_data, axis=0)
img_data = preprocess_input(img_data)



vgg16_feature = model.predict(img_data)

#print(vgg16_feature.shape)
square = 8
ix = 1
for _ in range(square):
	for _ in range(square):
		# specify subplot and turn of axis
		ax = plt.subplot(square, square, ix)
		ax.set_xticks([])
		ax.set_yticks([])
		# plot filter channel in grayscale
		plt.imshow(vgg16_feature[0, :, :, ix-1], cmap='gray')
		ix += 1
# show the figure
plt.show()

#plt.imshow(vgg16_feature[0,:,:,510])
#plt.show()