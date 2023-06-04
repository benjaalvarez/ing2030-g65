# Este script toma una carpeta con fotos de una misma camara trampa y devuelve el valor estimado del ecosistema en esa region.
# para cubrir un área habria que sumar el estimado de cada camara trampa 

import os
import cv2 as cv
from tensorflow import keras as krs
import numpy as np
import matplotlib.pyplot as plt

class_names = ['Not Animal-Plane', 'Not Animal-Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Not Animal Ship', 'Not Animal-Truck'] ##insertar output labels de la Red Neuronal


# recibe una carpeta con imagenes, cada imagen tiene el formato dd/mm/aa_hh:mm:ss. Asumo que todas las imagenes son de un misma locacion
read_data = {} #diccionario vacio

directory = r"C:\Users\javie\Desktop\Innova\Images"
id_ = 0

for file_name in os.listdir(directory):
    imagen = {}
    if file_name.endswith('.jpg'):
        #leer imagen
        img = cv.imread(directory+'\\' + file_name)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        imagen['data'] = img
        #extraer metadata
        meta = file_name.split('_')
        imagen['meta'] = meta
        read_data[str(id_)] = imagen #agregar al diccionario
        id_ += 1 #siguiente id    



#Analisis
model = krs.models.load_model('image_classifier_1.model') 

for key  in read_data.keys():
    imagen = read_data[key]
    img = imagen['data']
    prediction = model.predict(np.array([img]) / 255)
    index = np.argmax(prediction)
    imagen['Label'] = class_names[index]

valor = 0
estimados = dict([
    ('Dog', 1),
    ('Frog', 8),
    ('Cat', 2),
    ('Deer', 15),
    ('Bird', 9),
    ('Horse', 6)
])

for key in read_data.keys():
    imagen = read_data[key]
    label = imagen['Label']
    if label in estimados.keys():
        valor += estimados[label]
    else:
        valor += 0 #redundante, pero me ayuda a leerlo mejor

print(f"El valor estimado de la biodiversidad en este ecosistema es ${valor}")

        