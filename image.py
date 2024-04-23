import sys
from ultralytics import YOLO
from util import (obtenerTextoDesdeImagen, detectarVehiculos)
import cv2
import matplotlib.pyplot as plt


# Load Models
license_plate_detector = YOLO('datasets/models/license_plate_detector.pt')

def zoom_at(img, zoom, coord=None):
    # Translate to zoomed coordinates
    h, w, _ = [zoom * i for i in img.shape]

    if coord is None:
        cx, cy = w / 2, h / 2
    else:
        cx, cy = [zoom * c for c in coord]

    img = cv2.resize(img, (0, 0), fx=zoom, fy=zoom)
    return img

find_index = 0

name = 'auto-10'

dirArchivo = 'datasets/for_detections/images/' + name + '.png'
image_real = cv2.imread(dirArchivo)
h, w, c = image_real.shape
aspect_ratio = w / h

image = zoom_at(image_real, 1.2)

vehicle_detected = detectarVehiculos(image, crop=True, thickness=3)
if len(vehicle_detected) > 0:
    vehicle_detected = vehicle_detected[0]
    license_plates = license_plate_detector(vehicle_detected, verbose=False)[0]
    license_plates = license_plates.boxes.data.tolist()
    if(len(license_plates) > 0):
        find_index = find_index + 1
        x1, y1, x2, y2, lpscore, lpclass_id = license_plates[0]

        cv2.rectangle(vehicle_detected, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
        frame = cv2.cvtColor(vehicle_detected, cv2.COLOR_BGR2RGB)
        #cv2.imwrite('datasets/detections/car.png', frame)
        plate_frame = vehicle_detected[int(y1):int(y2), int(x1): int(x2), :]

        plate = obtenerTextoDesdeImagen(plate_frame)
        if(plate == ""):
            plate = "No se pudo detectar la placa"

        plt.subplot(1, 1, find_index)
        plt.xticks([]), plt.yticks([])
        plt.ylabel(name.replace('-',' '))
        plt.xlabel(plate)
        plt.imshow(frame)

plt.suptitle('Placa de carro detectado')
plt.savefig('datasets/detections/car.png')
plt.show()