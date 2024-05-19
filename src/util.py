import cv2
import pytesseract
import easyocr
import keras_ocr
import datetime

from ultralytics import YOLO
from sort.sort import *
pipeline = keras_ocr.pipeline.Pipeline()
reader = easyocr.Reader(['en'], gpu=False)
if os.name != 'posix':
    pytesseract.pytesseract.tesseract_cmd = r'E:\Tesseract-OCR\tesseract'
coco_model = YOLO('../datasets/models/yolov8n.pt')
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
    #texto = texto.upper()
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
    if parte_inicial[0] in ['8']:
        parte_inicial = 'B' + parte_inicial[1:3]

    """
    dict = {'D': 0, 'B': 8, 'o':0 }
    for key, value in dict.items():
        parte_final = parte_final.replace(key, str(value))
    """

    if not (isInteger(parte_final)):
        return ''
    text = parte_inicial + '-' + parte_final
    text = text.upper()
    return text
def _kerasocr(image):
    dd = keras_ocr.tools.read(image)
    prediction = pipeline.recognize([dd])[0]
    text = ''
    for dp in prediction:
        p = dp[0]
        if (p == 'peru' or p == 'pero' or p == 'per' or p == 'ped'):
            continue
        if not (len(p) == 0 or len(p) < 6 or len(p) > 7):
            text = p
    text = limpiarPlaca(text)
    return text
def _pytesseract(image):
    text = pytesseract.image_to_string(image, config="--psm 10 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ-0123456789")
    text = text.upper().strip().replace(' ', '').replace('\n', '')
    if len(text) == 0 or len(text) < 6 or len(text) > 7:
        return '';
    text = limpiarPlaca(text)
    return text
def _easyocr(image):
    detections = reader.readtext(image)
    _text = ''
    for detection in detections:
        bbox, _text, score = detection
        _text = _text.upper().strip().replace(' ', '')
        if len(_text) in [6, 7]:
            break
    text = _text
    text = limpiarPlaca(text)
    return text
def obtenerTextoDesdeImagen(image, getArray = False):
    if(getArray):
        t1 = _kerasocr(image)
        t2 = _pytesseract(image)
        t3 = _easyocr(image)
        return [t1, t2, t3]
    else:
        text = _pytesseract(image)
        if text == '':
            text = _easyocr(image)
        return text
def calculatePlate(plates):
    f = [{}, {}, {}, {}, {}, {}, {}]
    for item in plates:
        b = 0
        for char in item:
            if not char in f[b]:
                f[b][char] = 1
            else:
                f[b][char] += 1
            b += 1
    text = ''
    a = 0
    for items in f:
        f[a] = dict(sorted(items.items(), key=lambda x: x[1], reverse=True))
        keys = list(f[a].keys())
        text += keys[0]
        a += 1
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
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), thickness)
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
    return total_videos-1
def resize_frame(frame, nuevo_ancho, nuevo_alto):
    #alto, ancho, c = frame.shape
    #aspect_ratio = ancho / alto
    #nuevo_ancho = int(nuevo_alto * aspect_ratio)
    frame = cv2.resize(frame, (nuevo_ancho, nuevo_alto))
    return frame
    #return nuevo_alto, nuevo_ancho, aspect_ratio, frame
def zoom_at(img, zoom, coord=None):
    # Translate to zoomed coordinates
    h, w, _ = [zoom * i for i in img.shape]

    if coord is None:
        cx, cy = w / 2, h / 2
    else:
        cx, cy = [zoom * c for c in coord]

    img = cv2.resize(img, (0, 0), fx=zoom, fy=zoom)
    return img

def add_transparent_image(background, foreground, x_offset=None, y_offset=None):
    bg_h, bg_w, bg_channels = background.shape
    fg_h, fg_w, fg_channels = foreground.shape

    assert bg_channels == 3, f'background image should have exactly 3 channels (RGB). found:{bg_channels}'
    assert fg_channels == 4, f'foreground image should have exactly 4 channels (RGBA). found:{fg_channels}'

    # center by default
    if x_offset is None: x_offset = (bg_w - fg_w) // 2
    if y_offset is None: y_offset = (bg_h - fg_h) // 2

    w = min(fg_w, bg_w, fg_w + x_offset, bg_w - x_offset)
    h = min(fg_h, bg_h, fg_h + y_offset, bg_h - y_offset)

    if w < 1 or h < 1: return

    # clip foreground and background images to the overlapping regions
    bg_x = max(0, x_offset)
    bg_y = max(0, y_offset)
    fg_x = max(0, x_offset * -1)
    fg_y = max(0, y_offset * -1)
    foreground = foreground[fg_y:fg_y + h, fg_x:fg_x + w]
    background_subsection = background[bg_y:bg_y + h, bg_x:bg_x + w]

    # separate alpha and color channels from the foreground image
    foreground_colors = foreground[:, :, :3]
    alpha_channel = foreground[:, :, 3] / 255  # 0-255 => 0.0-1.0

    # construct an alpha_mask that matches the image shape
    alpha_mask = np.dstack((alpha_channel, alpha_channel, alpha_channel))

    # combine the background with the overlay image weighted by alpha
    composite = background_subsection * (1 - alpha_mask) + foreground_colors * alpha_mask

    # overwrite the section of the background image that has been updated
    background[bg_y:bg_y + h, bg_x:bg_x + w] = composite

def draw_security_camera_design(frame, ancho, alto, current_frame, place, numero_camara, tipo_camara, logo, prev_frame_time, padding = 30, tamano = 150, color = (255, 255, 255)):
    ### Esquina izquierda superior
    cv2.line(frame, (padding, padding), (padding, padding + tamano), color, 1)
    cv2.line(frame, (padding, padding), (padding + tamano, padding), color, 1)

    ### Esquina derecha superior
    cv2.line(frame, (ancho - padding, padding), (ancho - padding - tamano, padding), color, 1)
    cv2.line(frame, (ancho - padding, padding), (ancho - padding, padding + tamano), color, 1)

    ### Esquina izquierda inferior
    cv2.line(frame, (padding, alto - padding), (padding, alto - tamano - padding), color, 1)
    cv2.line(frame, (padding, alto - padding), (padding + tamano + padding, alto - padding), color, 1)

    ### Esquina derecha inferior
    cv2.line(frame, (ancho - padding, alto - padding),
             (ancho - padding - tamano, alto - padding), color, 1)
    cv2.line(frame, (ancho - padding, alto - padding),
             (ancho - padding, alto - padding - tamano), color, 1)

    ## Agregamos el número de cámara
    nombre_estacionamiento = place['place_name']
    cv2.putText(frame, "Camara " + str(numero_camara), (ancho - padding - 110, padding + 25), font, 0.5, color, 1,
                cv2.LINE_AA)
    cv2.putText(frame, nombre_estacionamiento + " - " + tipo_camara.title(),
                (padding + 10, padding + 25), font, 0.5, color, 1, cv2.LINE_AA)

    cv2.putText(frame, 'Espacio: ' + str(place['free']) + '/' + place['spaces'],
                (padding + 10, padding + 45), font, 0.5, color, 1, cv2.LINE_AA)


    cv2.putText(frame, place['place_address'],
                (padding + 15, alto - 10), font, 0.4, color, 1, cv2.LINE_AA)

    if (current_frame % 10 == 0):
        red = (0, 0, 205)
    else:
        red = (0, 0, 255)

    cv2.circle(frame, (ancho - padding - 20, padding + 20), 5, red, -1)

    ## Agregamos la fecha y el tiempo
    now = datetime.datetime.now()
    year = str(now.year)
    month = str(now.month)
    if (len(month) == 1):
        month = '0' + month
    day = str(now.day)
    if (len(day) == 1):
        day = '0' + day
    hour = str(now.hour)
    if (len(hour) == 1):
        hour = '0' + hour
    minute = str(now.minute)
    if (len(minute) == 1):
        minute = '0' + minute
    second = str(now.second)
    if (len(second) == 1):
        second = '0' + second
    microsecond = str(now.microsecond)
    date_now = day + '/' + month + '/' + year
    time_now = hour + ':' + minute + ':' + second + '.' + microsecond
    cv2.rectangle(frame, (padding + 15, alto - padding - 50), (padding + 138, alto - padding - 15),
                  (255, 255, 255), -1)
    cv2.rectangle(frame, (ancho - padding - 15, alto - padding - 15),
                  (ancho - padding - 172, alto - padding - 50), (255, 255, 255), -1)
    cv2.putText(frame, date_now, (padding + 25, alto - padding - 28), font, 0.5, (20, 20, 20), 1, cv2.LINE_AA)
    cv2.putText(frame, time_now, (ancho - padding - 160, alto - padding - 28), font, 0.5, (20, 20, 20), 1, cv2.LINE_AA)

    ## Calculamos y agregamos los FPS
    new_frame_time = time.time()
    fps = int(1 / (new_frame_time - prev_frame_time))
    prev_frame_time = new_frame_time
    cv2.putText(frame, "FPS: " + str(fps), (ancho - padding - 95, padding + 45), font, 0.5, (255, 255, 255), 1,
                cv2.LINE_AA)

    ## Dibujamos el logo
    x1 = int((ancho / 2) - 40)
    y1 = 10
    frame[int(y1):int(y1) + logo.shape[0], int(x1):int(x1) + logo.shape[1]] = logo
    #add_transparent_image(frame, logo, int((ancho / 2) - 40), 5)
    return prev_frame_time, nombre_estacionamiento, frame
