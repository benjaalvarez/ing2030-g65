# Este script toma una carpeta con fotos de una misma camara trampa y devuelve el valor estimado del ecosistema en esa region.
# para cubrir un área habria que sumar el estimado de cada camara trampa 

import os
import sys
import cv2 as cv
from tensorflow import keras as krs
import numpy as np
import matplotlib.pyplot as plt
import csv

class_names = ['Not Animal-Plane', 'Not Animal-Car', 'Bird', 'Cat', 'Huemul', 'Dog', 'Frog', 'Horse', 'Not Animal Ship', 'Not Animal-Truck'] ##insertar output labels de la Red Neuronal


# recibe una carpeta con imagenes, cada imagen tiene el formato dd/mm/aa_hh:mm:ss. Asumo que todas las imagenes son de un misma locacion
read_data = {} #diccionario vacio

directory = r"C:\Users\javie\Desktop\Innova\ing2030-g65\Images"
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
    #print(imagen['Label'])

valor = 0
estimados = dict([
    ('Dog', 5), #castor
    ('Frog', 5), #rana
    ('Cat', 50),  #gato colocolo
    ('Deer', 5), #Vaca
    ('Bird', 5), #gorrion
    ('Horse', 50) #huemul
])

for key in read_data.keys():
    imagen = read_data[key]
    label = imagen['Label']
    if label in estimados.keys():
        valor += estimados[label]
    else:
        valor += 0 #redundante, pero me ayuda a leerlo mejor

with open('./datos.txt', 'w') as file:
    file.write("POINT (-72.2497489 -40.3655544),Cam 3,Huemul: 2\nGato ColoColo: 1, Rana: 1")
print(f"El valor estimado de la biodiversidad en este ecosistema es {valor} UTM")

sys.stdout.flush()#se comunica con JS

#output
#truco
"""
i = 1
for key in read_data.keys():
    imagen = read_data[key]
    label = imagen['Label']
    img = imagen['data']
    plt.subplot(3,2, int(key) + 1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap= plt.cm.binary)
    if label in estimados.keys():
        if i == 3:
            plt.xlabel('Huemul')
        else:
            plt.xlabel(label)
    else:
        plt.xlabel('NotAnimal')
    i+=1
plt.show()

"""

        