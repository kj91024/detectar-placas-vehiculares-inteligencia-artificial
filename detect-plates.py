import sys
from ultralytics import YOLO
from util import (obtenerTextoDesdeImagen, detectarVehiculos, zoom_at)
import cv2
import matplotlib.pyplot as plt

# Load Models
license_plate_detector = YOLO('datasets/models/license_plate_detector.pt')

names = [
    'auto-1',
    'auto-2',
    'auto-3',
    'auto-4',
    'auto-5',
    'auto-6',
    'auto-7',
    'auto-8',
    'auto-9',
    'auto-10',
    'auto-11',
    'auto-12',
    'auto-13',
    'auto-14',
    'auto-15'
]

if(len(names) > 17):
    sys.exit("Solo puedes tener 16 imágenes")

img_array = autos_array = placas_array = []
titulos_array = []
find_index = 0
for x in range(0, len(names)):
    name = names[x]
    dirArchivo = 'datasets/for_detections/images/' + name + '.png'
    image_real = cv2.imread(dirArchivo)
    h, w, c = image_real.shape
    aspect_ratio = w / h

    image = zoom_at(image_real, 1.2)

    vehicle_detected = detectarVehiculos(image, crop=True)
    if len(vehicle_detected) > 0:
        vehicle_detected = vehicle_detected[0]
        cv2.imwrite('datasets/detections/images/' + name + '.png', vehicle_detected)
        license_plates = license_plate_detector(vehicle_detected, verbose=False)[0]
        license_plates = license_plates.boxes.data.tolist()
        if(len(license_plates) > 0):
            find_index = find_index + 1
            x1, y1, x2, y2, lpscore, lpclass_id = license_plates[0]
            frame = vehicle_detected[int(y1):int(y2), int(x1): int(x2), :]
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cv2.imwrite('datasets/detections/plates/' + name + '.png', frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

            plate = obtenerTextoDesdeImagen(frame)
            plt.subplot(4, 3, find_index)
            plt.xticks([]), plt.yticks([])
            plt.ylabel(name.replace('-',' '))
            plt.xlabel(plate)
            plt.imshow(frame)

plt.suptitle('Placas de carro detectadas')
plt.savefig('datasets/detections/plates.png')
plt.show()