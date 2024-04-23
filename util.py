import cv2
import pytesseract
import easyocr
from ultralytics import YOLO
from sort.sort import *
reader = easyocr.Reader(['en'], gpu=False)
#pytesseract.pytesseract.tesseract_cmd = r'E:\Tesseract-OCR\tesseract'
coco_model = YOLO('datasets/models/yolov8n.pt')
mot_tracker = Sort()

font = cv2.FONT_HERSHEY_SIMPLEX

def zoom(img, zoom, coord=None):
    # Translate to zoomed coordinates
    h, w, _ = [zoom * i for i in img.shape]

    if coord is None:
        cx, cy = w / 2, h / 2
    else:
        cx, cy = [zoom * c for c in coord]

    img = cv2.resize(img, (0, 0), fx=zoom, fy=zoom)
    """
    img = img[ int(round(cy - h/zoom * .5)) : int(round(cy + h/zoom * .5)),
               int(round(cx - w/zoom * .5)) : int(round(cx + w/zoom * .5)),
               : ]
    """
    return img
def isInteger(n):
    try:
        float(n)
    except ValueError:
        return False
    else:
        return float(n).is_integer()
def limpiarPlaca(texto):
    for t in [';', '*', ':', '[', ']', '(', ')', '{', '}', '/', '\\', '+', '~', '"', "'", '|', '$', '#', '%', '&', '=',
              '_', '@', '.',',',';',':','_','-','|','!','¡','?','¿','<','>','^']:
        texto = texto.replace(t, '-')

    parte_inicial = ''
    parte_final = ''
    if len(texto) == 6:
        parte_inicial = texto[:3]
        parte_final = texto[3:]
    elif len(texto) == 7:
        parte_inicial = texto[:3]
        parte_final = texto[4:]
    else:
        return ''

    if len(parte_inicial.replace('-', '')) < 3 or len(parte_final.replace('-', '')) < 3:
        return ''

    if parte_inicial[0] in ['2', '7']:
        parte_inicial = 'Z' + parte_inicial[1:3]

    dict = {'D': 0, 'B': 8}
    for key, value in dict.items():
        parte_final = parte_final.replace(key, str(value))

    if not (isInteger(parte_final)):
        return ''
    text = parte_inicial + '-' + parte_final
    return text
def obtenerTextoDesdeImagen(image):
    text = pytesseract.image_to_string(image,
                                       config="--psm 10 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ-0123456789")
    text = text.upper().strip().replace(' ', '').replace('\n', '')

    if len(text) == 0 or len(text) < 6 or len(text) > 7:
        detections = reader.readtext(image)
        _text = ''
        for detection in detections:
            bbox, _text, score = detection
            _text = _text.upper().strip().replace(' ', '')
            if len(_text) in [6, 7]:
                break
        text = _text

    text = limpiarPlaca(text)
    # if len(text) != 0:
    #    print("text: ", text)
    return text
def pantallaDeNoDeteccion(frame, nuevo_ancho, nuevo_alto, padding=10):
    # Dibujamos en la pantalla de no detección
    cv2.rectangle(frame, (0, nuevo_alto-50), (nuevo_ancho, nuevo_alto), (0, 0, 0), -1)
    #cv2.rectangle(frame, (int(padding), int(padding)), (int(nuevo_ancho - padding), int(nuevo_alto - padding)), (255, 255, 255), 1)
    cv2.putText(frame, 'No se puede determinar', (int(nuevo_ancho / 2) - 250, int(nuevo_alto - 15)), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
def obtenerCarro(license_plate, vehicle_track_ids):
    """
    Retrieve the vehicle coordinates and ID based on the license plate coordinates.

    Args:
        license_plate (tuple): Tuple containing the coordinates of the license plate (x1, y1, x2, y2, score, class_id).
        vehicle_track_ids (list): List of vehicle track IDs and their corresponding coordinates.

    Returns:
        tuple: Tuple containing the vehicle coordinates (x1, y1, x2, y2) and ID.
    """
    x1, y1, x2, y2, score, class_id = license_plate

    foundIt = False
    for j in range(len(vehicle_track_ids)):
        xcar1, ycar1, xcar2, ycar2, car_id = vehicle_track_ids[j]

        if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
            car_indx = j
            foundIt = True
            break

    if foundIt:
        return vehicle_track_ids[car_indx]

    return -1, -1, -1, -1, -1
def detectarVehiculos(frame, crop=False, thickness=1):
    vehicles = [2]
    detections = coco_model(frame, verbose=False)[0]
    frame_vehicle_detections = []
    for detection in detections.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = detection
        if int(class_id) in vehicles:
            frame_vehicle_detections.append([x1, y1, x2, y2, score])
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), thickness)
            if crop == True:
                return [frame[int(y1):int(y2), int(x1): int(x2), :]]
    return frame_vehicle_detections
def trackearVehiculos(vehiculos_detectados):
    # track vehicles
    matriz = np.asarray(vehiculos_detectados)
    track_ids = mot_tracker.update(matriz)
    return track_ids
def procesarPlacaVehicular(frame):
    frame_plate = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY);
    return frame_plate
def dibujarPlaca(frame, x1, y1, x2, y2, text="", padding = 0):
    if(text != ""):
        cv2.rectangle(frame, (int(x1 + padding), int(y1 + padding)), (int(x2 - padding), int(y2 - padding)), (0, 255, 0), 3)
        #cv2.rectangle(frame, (int(x1 + padding), int(y1 - 30 + padding)), (int(x1 + 100), int(y1 + padding - 2)), (0, 255, 0), -1)
        #cv2.putText(frame, text, (int(x1 + 5 + padding), int(y1 - 10 + padding)), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    else:
        cv2.rectangle(frame, (int(x1 + padding), int(y1 + padding)), (int(x2 - padding), int(y2 - padding)), (255, 255, 255), 2)
        #cv2.rectangle(frame, (int(x1 + padding), int(y1 - 30 + padding)), (int(x1 + 115), int(y1 + padding - 2)), (255, 255, 255), -1)
        #cv2.putText(frame, 'Esta borroso', (int(x1 + 5 + padding), int(y1 - 10 + padding)), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

def totalVideos(ruta):
    total_videos = 0
    # Iterate directory
    for path in os.listdir(ruta):
        # check if current path is a file
        if os.path.isfile(os.path.join(ruta, path)):
            total_videos += 1
    return total_videos

def zoom_at(img, zoom, coord=None):
    # Translate to zoomed coordinates
    h, w, _ = [zoom * i for i in img.shape]

    if coord is None:
        cx, cy = w / 2, h / 2
    else:
        cx, cy = [zoom * c for c in coord]

    img = cv2.resize(img, (0, 0), fx=zoom, fy=zoom)
    return img