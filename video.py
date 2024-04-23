from ultralytics import YOLO
import cv2
from sort.sort import *
from util import obtenerTextoDesdeImagen, pantallaDeNoDeteccion, obtenerCarro

# load models
mot_tracker = Sort()
coco_model = YOLO('datasets/models/yolov8n.pt')
license_plate_detector = YOLO('datasets/models/license_plate_detector.pt')

font = cv2.FONT_HERSHEY_SIMPLEX
# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('datasets/for_detections/videos/carros4.mp4')
property_id = int(cv2.CAP_PROP_FRAME_COUNT)
total_frames = int(cv2.VideoCapture.get(cap, property_id))
 
# Check if camera opened successfully
if (cap.isOpened() == False):
    print("Error opening video stream or file")

# 2 car
# 3 motorcycle
# 5 bus
# 7 truck
vehicles = [2]
#vehicles = [0]
#coco_model.predict(source="for_training/images/", save=False, classes=vehicles)
#exit()
#license_plate_detector.predict(source="for_training/images/", save=False, classes=[0])
#exit()

detalle_placas = {}
# Read until video is completed
frame_nmr = -1
p = 0
text = None
lpx1, lpy1, lpx2, lpy2, lpscore, lpclass_id = 0,0,0,0,0,0
while(cap.isOpened()):
    frame_nmr += 1
    ret, frame = cap.read()

    if frame is None or ret == False:
        break

    alto, ancho, c = frame.shape
    aspect_ratio = ancho / alto

    # Formato HD 720p
    # nuevo_ancho = 1280
    # nuevo_alto = int(nuevo_ancho / aspect_ratio)

    nuevo_alto = 720
    nuevo_ancho = int(nuevo_alto * aspect_ratio)

    frame = cv2.resize(frame, (nuevo_ancho, nuevo_alto))

    #print('video_frame: ', frame_nmr, '/', total_frames, ', aspect_ratio: ', aspect_ratio)

    if frame_nmr % 5 == 0 and ret == True:

        # Detección de vehiculos
        detections = coco_model(frame, verbose=False)[0]
        frame_vehicle_detections = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in vehicles:
                frame_vehicle_detections.append([x1, y1, x2, y2, score])
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), 1)
                #vehicle_crop = frame[int(y1):int(y2), int(x1): int(x2), :]

        # Solo detectamos las plates si encontramos cars
        if len(frame_vehicle_detections) > 0:

            # track vehicles
            matriz = np.asarray(frame_vehicle_detections)
            if (len(matriz) == 0):
                continue
            track_ids = mot_tracker.update(matriz)

            # Detección de plates
            license_plates = license_plate_detector(frame, verbose=False)[0]
            frame_plates = []
            i_plates = 0
            for license_plate in license_plates.boxes.data.tolist():
                i_plates += 1
                lpx1, lpy1, lpx2, lpy2, lpscore, lpclass_id = license_plate

                # assign license plate to car
                xcar1, ycar1, xcar2, ycar2, car_id = obtenerCarro(license_plate, track_ids)

                w = lpx2 - lpx1
                h = lpy2 - lpy1
                if h == 0:
                    break

                aspect_ratio = w / h
                if aspect_ratio > 2.4 or h > w:
                    break

                padding = 0
                frame_plates_crop = frame[int(lpy1 + padding):int(lpy2 + padding), int(lpx1 - padding): int(lpx2 - padding), :]
                cv2.imwrite('placas_detectadas/placa' + frame_nmr + '.jpg', frame_plates_crop)
                print('placas_detectadas/placa' + frame_nmr + '.jpg')
                _alto, _ancho, c = frame_plates_crop.shape
                if(_alto == 0 or _ancho == 0):
                    break

                frame_plates_crop_gray = cv2.cvtColor(frame_plates_crop, cv2.COLOR_BGR2GRAY)
                text = obtenerTextoDesdeImagen(frame_plates_crop_gray)

                if len(text) == 7:
                    text = text.strip().replace('?', '')
                    cv2.rectangle(frame, (int(lpx1 + padding), int(lpy1 + padding)), (int(lpx2 - padding), int(lpy2 - padding)), (0, 255, 0), 2)
                    cv2.rectangle(frame, (int(lpx1 + padding), int(lpy1 - 30 + padding)), (int(lpx1 + 100), int(lpy1 + padding - 2)), (0, 255, 0), -1)
                    cv2.putText(frame, text, (int(lpx1 + 5 + padding), int(lpy1 - 10 + padding) ), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

                    print("text: ", text)

                    car_id = str(car_id)
                    if not(car_id in detalle_placas):
                        detalle_placas[car_id] = {}

                    if text in detalle_placas[car_id]:
                        detalle_placas[car_id][text] = detalle_placas[car_id][text] + 1
                    else:
                        detalle_placas[car_id][text] = 1
                else:
                    cv2.rectangle(frame, (int(lpx1 + padding), int(lpy1 + padding)), (int(lpx2 - padding), int(lpy2 - padding)), (255, 255, 255), 2)
                    cv2.rectangle(frame, (int(lpx1 + padding), int(lpy1 - 30 + padding)), (int(lpx1 + 115), int(lpy1 + padding - 2)), (255, 255, 255), -1)
                    cv2.putText(frame, 'Esta borroso', (int(lpx1 + 5 + padding), int(lpy1 - 10 + padding)), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        else:
            pantallaDeNoDeteccion(frame, nuevo_ancho, nuevo_alto, 20)

        # Display the resulting frame
        cv2.imshow('Frame', frame)

    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

print(detalle_placas)
# When everything done, release the video capture object
cap.release()
# Closes all the frames
cv2.destroyAllWindows()