# Importar numpy y matplotlib.pyplot
import numpy as np
import matplotlib.pyplot as plt

# Importar OpenCV
import cv2 as cv

# Importar elementos necesarios de Tensorflow
from tensorflow import keras  as krs

(training_images, training_labels), (testing_images, testing_labels) = krs.datasets.cifar10.load_data()

training_images, testing_images = training_images / 255, testing_images / 255
class_names = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

for image in range(16):
    plt.subplot(4,4, image+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(training_images[image], cmap=plt.cm.binary)
    plt.xlabel(class_names[training_labels[image][0]])

plt.show()
print(len(training_images))
print(len(testing_images))

"""
## reducir cantidad de imagenes que entran a la red
training_images = training_images[:20000]
training_labels = training_labels[:20000]
testing_images = testing_images[:4000]
testing_labels = testing_labels[:4000]
"""
"""
#build Neural Network
model = krs.models.Sequential()

model.add(krs.layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)))
model.add(krs.layers.MaxPooling2D((2,2)))
model.add(krs.layers.Conv2D(64, (3,3), activation='relu'))
model.add(krs.layers.MaxPooling2D((2,2)))
model.add(krs.layers.Conv2D(64, (3,3), activation='relu'))
model.add(krs.layers.Flatten())
model.add(krs.layers.Dense(64, activation='relu'))
model.add(krs.layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=10, validation_data=(testing_images,testing_labels))

#Guardar modelo
loss, accuracy = model.evaluate(testing_images, testing_labels)
print(f"Loss: {loss}")
print(f"Accuracy: {accuracy}")

model.save('image_classifier_1.model')
"""
model = krs.models.load_model('image_classifier_1.model')

img= cv.imread('horse1.jpg')
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
#plt.imshow(img, cmap=plt.cm.binary)

prediction = model.predict(np.array([img]) / 255)
index = np.argmax(prediction)

print(f"Prediction for horse.jpg is {class_names[index]}")

#######

img= cv.imread('plane1.jpg')
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
prediction = model.predict(np.array([img]) / 255)
index = np.argmax(prediction)
print(f"Prediction for plane.jpg is {class_names[index]}")

img= cv.imread('cat1.jpg')
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
prediction = model.predict(np.array([img]) / 255)
index = np.argmax(prediction)
print(f"Prediction for cat.jpg is {class_names[index]}")

img= cv.imread('truck1.jpg')
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
prediction = model.predict(np.array([img]) / 255)
index = np.argmax(prediction)
print(f"Prediction for truck.jpg is {class_names[index]}")