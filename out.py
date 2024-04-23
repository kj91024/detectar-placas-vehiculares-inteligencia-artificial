from ultralytics import YOLO
from util import (obtenerTextoDesdeImagen, pantallaDeNoDeteccion, obtenerCarro, detectarVehiculos,
                  trackearVehiculos, procesarPlacaVehicular, dibujarPlaca, totalVideos)
from tkinter import *
from PIL import Image, ImageTk
import cv2
import imutils
import os
import requests
# Load Models
license_plate_detector = YOLO('datasets/models/license_plate_detector.pt')

fps = 30 # Frames por segundo
segundos = 1 # Segundos de análisis
analisis_frame = int(fps * segundos) # Tiempo de analisis

total_videos = totalVideos('datasets/for_detections/videos/out')
is_video = False
index_video = 1
detalle_placas = {}
frame_nmr = _frame_nmr = -1
total_frames = -1
p = 0
text = None
lpx1, lpy1, lpx2, lpy2, lpscore, lpclass_id = 0,0,0,0,0,0
def video_stream():
    global frame_nmr, total_frames, cap, is_video, index_video, _frame_nmr
    frame_nmr += 1

    # Mostramos el video si llegamos al frame seleccionado
    if frame_nmr == ((fps * index_video) + (fps * (index_video-1))) and index_video <= total_videos:
        is_video = True
        _frame_nmr = ((fps * index_video) + (fps * (index_video-1)))
        video_path = 'datasets/for_detections/videos/out/' + str(index_video) + '.mp4'
        cap = cv2.VideoCapture(video_path)
        property_id = int(cv2.CAP_PROP_FRAME_COUNT)
        total_frames = int(cv2.VideoCapture.get(cap, property_id))
        #print("total: ", total_frames)
        if (cap.isOpened() == False):
            print("Error opening video file: " + video_path)

    # Retomamos la grabacion por cámara
    if is_video == True and frame_nmr > _frame_nmr + analisis_frame:
        is_video = False
        index_video += 1
        cap = cv2.VideoCapture(0)
        property_id = int(cv2.CAP_PROP_FRAME_COUNT)
        total_frames = int(cv2.VideoCapture.get(cap, property_id))
        if (cap.isOpened() == False):
            print("Error opening video stream")

    ret, frame = cap.read()

    if frame is None or ret == False:
        return

    # Resize video
    alto, ancho, c = frame.shape
    aspect_ratio = ancho / alto
    nuevo_alto = 720
    nuevo_ancho = int(nuevo_alto * aspect_ratio)
    frame = cv2.resize(frame, (nuevo_ancho, nuevo_alto))
    #print('video_frame: ', frame_nmr, '/', total_frames, ', aspect_ratio: ', aspect_ratio)

    # frame_nmr % 2 == 0 and
    if ret == True:
        # Detección de vehiculos
        frame_vehicle_detections = detectarVehiculos(frame)

        # Solo detectamos las plates si encontramos cars
        if len(frame_vehicle_detections) > 0:

            # track vehicles
            track_ids = trackearVehiculos(frame_vehicle_detections)

            # Detección de plates
            license_plates = license_plate_detector(frame, verbose=False)[0]
            frame_plates = []
            i_plates = 0
            for license_plate in license_plates.boxes.data.tolist():
                i_plates += 1
                lpx1, lpy1, lpx2, lpy2, lpscore, lpclass_id = license_plate

                # Assign license plate to car
                xcar1, ycar1, xcar2, ycar2, car_id = obtenerCarro(license_plate, track_ids)

                w = lpx2 - lpx1
                h = lpy2 - lpy1
                if h == 0:
                    break

                aspect_ratio = w / h
                if aspect_ratio > 2.4 or h > w:
                    break

                padding = 0
                frame_plates_crop = frame[int(lpy1 + padding) : int(lpy2 + padding), int(lpx1 - padding) : int(lpx2 - padding), :]
                #cv2.imwrite('placas_detectadas/placa-' + str(index_video) + '-' + str(frame_nmr) + '.jpg', frame_plates_crop)

                _alto, _ancho, c = frame_plates_crop.shape
                if(_alto == 0 or _ancho == 0):
                    break

                frame_plate = procesarPlacaVehicular(frame_plates_crop)
                text = obtenerTextoDesdeImagen(frame_plate)
                if len(text) == 7:
                    dibujarPlaca(frame, lpx1, lpy1, lpx2, lpy2, text)

                    car_id = str(car_id + index_video)
                    if not(car_id in detalle_placas):
                        detalle_placas[car_id] = {'cantidad': 1}
                    else:
                        detalle_placas[car_id]['cantidad'] = detalle_placas[car_id]['cantidad'] + 1

                    if text in detalle_placas[car_id]:
                        detalle_placas[car_id][text] = detalle_placas[car_id][text] + 1
                    else:
                        detalle_placas[car_id][text] = 1

                    #print('carro ' + car_id + ':', detalle_placas[car_id]['cantidad'])
                    #if detalle_placas[car_id]['cantidad'] == 5:
                    _frame_plates_crop = cv2.resize(frame_plate, (300, int(300 / 1.9)))
                    cv2image = cv2.cvtColor(_frame_plates_crop, cv2.COLOR_BGR2RGBA)
                    img = Image.fromarray(cv2image)
                    imgtk = ImageTk.PhotoImage(image=img)
                    seccionPlacaVehicular.imgtk = imgtk
                    seccionPlacaVehicular.configure(image=imgtk)
                else:
                    dibujarPlaca(frame, lpx1, lpy1, lpx2, lpy2)

                # print(str(frame_nmr) + '==' + str(_frame_nmr + analisis_frame) + ', ' + str(_frame_nmr + total_frames))

                # print(str( (car_id in detalle_placas) ) + ' and ' + str(frame_nmr == _frame_nmr + analisis_frame) + ', ' + str(frame_nmr == _frame_nmr + total_frames - 1))

                if ((car_id in detalle_placas) and (frame_nmr == _frame_nmr + analisis_frame or frame_nmr == _frame_nmr + total_frames - 1)):
                    # print(detalle_placas);
                    # print(detalle_placas[car_id])
                    # print(sorted(detalle_placas[car_id].items(), key=lambda x: x[1], reverse=True))
                    plaquitas = dict(sorted(detalle_placas[car_id].items(), key=lambda x: x[1], reverse=True))
                    _placa = ''
                    for item in plaquitas.keys():
                        if (item == 'cantidad'):
                            continue
                        _placa = item
                        break
                    response = requests.get('http://localhost:8080/salida/' + _placa)
                    print(response)
        else:
            pantallaDeNoDeteccion(frame, nuevo_ancho, nuevo_alto, 20)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        seccionCamara.imgtk = imgtk
        seccionCamara.configure(image=imgtk)

    seccionCamara.after(1, video_stream)

cap = cv2.VideoCapture(0)
if (cap.isOpened() == False):
    print("Error opening video stream")

root = Tk()
root.title('IA DETECCIÓN DE PLACA VEHICULAR EN EL INGRESO')
root.geometry('1280x720')

#"""
# Lugar donde se mostrara el video de los cars
app = Frame(root, bg="white", width=1280).grid()
seccionCamara = Label(app)
seccionCamara.grid(column=0, row=0, rowspan=7)
#"""

#"""
# Lugar donde será visualizado la plates vehiculares con zoom
imagenPlaca = cv2.imread('datasets/sample.png')
seccionPlacaVehicular = Label(root, text="Placa Vehicular", width=300, height=150)
seccionPlacaVehicular.place(x=960, y=20)
imagenPlaca = cv2.cvtColor(imagenPlaca, cv2.COLOR_BGR2RGB)
imagenPlaca = imutils.resize(imagenPlaca, width=300)
imagenPlaca = Image.fromarray(imagenPlaca)
imagenPlaca = ImageTk.PhotoImage(image=imagenPlaca)
seccionPlacaVehicular.configure(image=imagenPlaca)
seccionPlacaVehicular.image = imagenPlaca
#"""

video_stream()
root.mainloop()