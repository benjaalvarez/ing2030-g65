
import os
import cv2 as cv
from tensorflow import keras as krs
import numpy as np

class_names = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck'] ##insertar outputs de la Red Neuronal


# recibe una carpeta con imagenes, cada imagen tiene el formato dd/mm/aa-hh:mm:ss
read_data = {} #diccionario vacio

directory = '\Folder'
id_ = 0
for filename in os.listdir(directory):
    imagen = {}
    if filename.endswith('.jpg'):
        #leer imagen
        img = cv.imread(filename)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        imagen['data'] = img
        #extraer metadata
        meta = filename.split('-')
        imagen['meta'] = meta
        read_data[str(id_)] = imagen #agregar al diccionario
        id_ += 1 #siguiente id

#Analisis
model = krs.models.load_model('image_classifier_1.model') 

for imagen in read_data:
    img = imagen['data']
    prediction = model.predict(np.array([img]) / 255)
    index = np.argmax(prediction)
    imagen['Label'] = class_names[index]




        