#=============================================================
# Configuración base
#=============================================================

# Conexión con el servicio
servidor = 'http://localhost:8080/'
# Código del estacionamiento que nos proporciona la plataforma
token_enlace = '717a47734151413d'

# Número de la camara
numero_camara = 1
# Puede ser "ingreso" o "salida"
tipo_camara = 'salida'
# Medio que serán analizados "video" o "camara"
medio = 'video'

# Si escogiste video, entonces especifíca la ruta del vídeo de donde se encuentra
#video_path = "datasets/for_detections/videos/in/5.mp4"
video_path = "datasets/for_detections/videos/out/23.mp4"
# Umbral maximo de recuento de reconomiento de placas, (Para el algoritmo de lista en profundidad)
umbral = 30
# Tiempo de espera entre reconomientos entre segundos
tiempo_espera = 120

#=============================================================
# Código
#=============================================================

import requests
import json
import base64
try:
    url = servidor + 'info/' + token_enlace
    response = requests.get(url)
    data = json.loads(response.text)
    print("Conexión con el servidor: "+data['status'])
    if data['status'] == 'error':
        print("Problema: " + data['data'])

    place = data['data']
    place['free'] = int(place['free'])
except:
    print("El servidor que configuraste esta mal o esta apagado, porfavor revisalo")
    exit(1)

print("Cargando recursos...")
import cv2

from ultralytics import YOLO
from sort.sort import *
coco_model = YOLO('datasets/models/yolov8n.pt')
license_plate_detector = YOLO('datasets/models/license_plate_detector.pt')
mot_tracker = Sort()

import numpy as np
from src.util import (resize_frame, zoom, draw_security_camera_design,
                      obtenerTextoDesdeImagen, calculatePlate)

if medio == 'video':
    if os.path.isfile(video_path):
        print("Video cargado: " + video_path)
        resource = video_path
    else:
        print("El video no existe: " + video_path)
        exit(1)
elif medio == 'camara':
    resource = 0
else:
    print("El medio solo puede ser 'video' o 'camara'")
    exit(1)

blanco = (255, 255, 255)
amarillo = (0, 255, 255)
verde = (0, 255, 0)
rojo = (0, 0, 255)

font = cv2.FONT_HERSHEY_SIMPLEX
cap = cv2.VideoCapture(resource)
property_id = int(cv2.CAP_PROP_FRAME_COUNT)
total_frames = int(cv2.VideoCapture.get(cap, property_id))

logo = cv2.imread("assets/utp_logo.png")
logo = zoom(logo, 0.08)

# Check if camera opened successfully
if (cap.isOpened() == False):
    print("Error opening video stream or file")

# Formato HD 720p
nuevo_ancho = 1280
nuevo_alto = 720
# Formato SD 480
#nuevo_ancho = 854
#nuevo_alto = 480

current_frame = prev_frame_time = new_frame_time = 0
out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (nuevo_ancho, nuevo_alto))

lista_placas = []
historial_placas = access = {}
stop_recognition = False
timestamp_recognition = time.time()
nivel_confianza = 0

while (cap.isOpened()):
    current_frame += 1
    ret, frame = cap.read()

    if frame is None or ret == False:
        break

    # Formateamos el video a calidad HD
    frame_colour = resize_frame(frame, nuevo_ancho, nuevo_alto)
    frame_gray = cv2.cvtColor(frame_colour, cv2.COLOR_BGR2GRAY)
    frame = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)

    if not stop_recognition:
        # Detectamos los vehiculos
        vehicles = [2, 7, 5]
        vehicules_detected = coco_model(frame_colour, verbose=False)[0]

        for detection in vehicules_detected.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in vehicles:
                crop = frame_colour[int(y1):int(y2), int(x1): int(x2), :]
                frame[int(y1):int(y1) + crop.shape[0], int(x1):int(x1) + crop.shape[1]] = crop
                #cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (100, 100, 100), 1)
        
        # Detectamos las placas
        license_plates = license_plate_detector(frame, verbose=False)[0]

        """
        # Trackeamos las placas
        matriz = np.asarray(license_plates)
        track_ids = mot_tracker.update(matriz)
        print(track_ids)
        """

        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate
            w = x2 - x1
            h = y2 - y1
            if h == 0:
                break
            # Verificamos que tenga el aspect_ratio de una placa peruana
            # Las placas actuales son el estándar norteamericano de 6 × 12 pulgadas (152 × 300 mm)
            # aspect_ratio = 300/152 = 1.97368421053
            aspect_ratio = w / h
            if aspect_ratio > 2.4 or h > w:
                break

            crop = frame_colour[int(y1):int(y2), int(x1): int(x2), :]
            crop_gray = frame[int(y1):int(y2), int(x1): int(x2), :]
            frame[int(y1):int(y1) + crop.shape[0], int(x1):int(x1) + crop.shape[1]] = crop

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), amarillo, 1)

            if len(historial_placas) == 0 or (len(historial_placas) > 0 and list(historial_placas.values())[0] < umbral):
                crop_gray = zoom(crop_gray, 1.2)

                placas = obtenerTextoDesdeImagen(crop_gray, True)
                for placa in placas:
                    if(placa != ''):
                        if placa in historial_placas:
                            historial_placas[placa] += 1
                        else:
                            historial_placas[placa] = 1
                        lista_placas.append(placa)
                #print(historial_placas)

            if(len(historial_placas) > 0 and list(historial_placas.values())[0] == umbral):
                timestamp_recognition = time.time()
                stop_recognition = True
                last_frame_detected = frame.copy()

                keys = list(historial_placas.keys())
                placa = calculatePlate(lista_placas)
                url = servidor + tipo_camara + '/' + place['id'] + '/' + placa + '?token=' + token_enlace
                url += '&signature=' + base64.b64encode(url.encode("ascii")).decode("ascii")
                response = requests.get(url)
                if(response.status_code == 200):
                    access = json.loads(response.text)
                    access['placa'] = placa
                    historial_placas = {}
                    lista_placas = []
                else:
                    print("Hay problemas con la peticion al servidor: " + url)
                    exit(0)
            elif(len(historial_placas) > 0):

                historial_placas = dict(sorted(historial_placas.items(), key=lambda x: x[1], reverse=True))

                plates = list(historial_placas.keys())
                count = historial_placas[plates[0]]
                placa = calculatePlate(lista_placas)

                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), verde, 1)
                percentage_certainty = str(int(100*count/umbral)) + '%'

                # Colocamos la placa detectada
                cv2.rectangle(frame, (int(x1), int(y1 - 30)), (int(x1 + 85), int(y1 - 2)), verde, -1)
                cv2.putText(frame, placa, (int(x1 + 5), int(y1 - 10)), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

                cv2.rectangle(frame, (int(x1 + 85), int(y1 - 30)), (int(x1 + 135), int(y1 - 2)), amarillo, -1)
                cv2.putText(frame, percentage_certainty, (int(x1 + 95), int(y1 - 10)), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    else:
        text = access['placa']
        text += " " + access['message']
        if access['status'] == 'success':
            color = (0, 200, 0)
            color_texto = (0, 0, 0)

            if (tipo_camara == 'ingreso' and current_frame % 30 == 0):
                url = servidor + 'get-data' + '/' + place['id'] + '?token=' + token_enlace
                url += '&signature=' + base64.b64encode(url.encode("ascii")).decode("ascii")
                response = requests.get(url)
                if (response.status_code == 200):
                    status = json.loads(response.text)
                    if (status['step'] == 3):
                        #timestamp_recognition = time.time()
                        text = access['placa'] + " puede ingresar"
                        access['message'] = 'puede ingresar'
                        color = verde
                    elif(status['step'] == 1):
                        place['free'] -= 1
                        stop_recognition = False
                else:
                    print("Hay problemas con la peticion al servidor: " + url)
                    exit(0)
            w = int(len(text) * 10)
            if(tipo_camara == 'salida'):
                espera = 30
            else:
                espera = tiempo_espera
        elif access['status'] == 'error':
            color = (0, 128, 255)
            color_texto = (0, 0, 0)
            w = int(len(text) * 9)
            espera = 10
        elif access['status'] == 'danger':
            color = rojo
            color_texto = (255, 255, 255)
            w = int(len(text) * 9)
            espera = 10

        m = int(nuevo_ancho/2) - int(w/2)
        cv2.rectangle(frame, (m, nuevo_alto - 100), (m + w, nuevo_alto - 60), color, -1)
        cv2.putText(frame, text, (m + 12, nuevo_alto - 75), font, 0.5, color_texto, 1, cv2.LINE_AA)

        if(timestamp_recognition + espera < time.time()):
            stop_recognition = False

    # Graficamos el diseño de camara de seguridad
    prev_frame_time, nombre_estacionamiento, frame = draw_security_camera_design(frame, nuevo_ancho, nuevo_alto, current_frame, place, numero_camara, tipo_camara, logo, prev_frame_time)

    # Guardando video
    out.write(frame)

    # Mostramos la imagen
    cv2.imshow('Frame', frame)

    cv2.setWindowTitle('Frame', "Camara " + str(numero_camara) + ": " + nombre_estacionamiento + " - " + tipo_camara.title())
    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

# Solo guardamos el video en linux
if os.name == 'posix':
    print("Convirtiendo video grabado...")
    os.system("ffmpeg -hide_banner -loglevel error -y -i output.mp4 -vcodec libx264 video.mp4")
    print("Video listo...")