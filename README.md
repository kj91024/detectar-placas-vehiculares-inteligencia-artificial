# Instalación
    
    pip install -r requirements.txt

# Entrenamiento de modelo
## Datos para entrenar el modelo
### Para detectar carros
- https://public.roboflow.com/object-detection/vehicles-openimages/1
- https://universe.roboflow.com/plat-kendaraan/vehicle-and-license-plate
---
- (Video) [https://drive.google.com/file/d/1JbwLyqpFCXmftaJY1oap8Sa6KfjoWJta/view]
- [https://boxy-dataset.com/boxy/]
- [https://www.kaggle.com/datasets/sshikamaru/car-object-detection]

### Para detectar placas
- https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e/dataset/4/download
- https://public.roboflow.com/object-detection/license-plates-us-eu/3
- https://universe.roboflow.com/augmented-startups/vehicle-registration-plates-trudk
---
- [https://datasetninja.com/car-license-plate#download]
- [https://www.kaggle.com/datasets/andrewmvd/car-plate-detection?resource=download]
- [https://data.mendeley.com/datasets/nx9xbs4rgx/2]

### Para detectar letras
- https://github.com/pragatiunna/License-Plate-Number-Detection/blob/main/data.zip

## Solo para iniciar el entrenamiento
Todos los archivos deben estar dentro de una misma carpeta, de la siguiente manera:
    
    /data
        /for_training
            /cars    # Los datos de carros
            /plates  # Los datos de las placas vehiculares

### Entramiento en Google Colab
Recomendamos realizarlo en Google Colab para que sea más rápido y no malogres tu computadora, para ello deberas usar los siguientes comandos dentro de la plataforma:
    
    !cd /content && git clone https://github.com/kj91024/detectar-placas-vehiculares-inteligencia-artificial.git
---
    !pip install ultralytics
---
    import locale
    locale.getpreferredencoding = lambda: "UTF-8"
---
    !unzip /content/detectar-placas-vehiculares-inteligencia-artificial/datasets/for_training/cars/car-1.zip -d /content/detectar-placas-vehiculares-inteligencia-artificial/datasets/for_training/cars/car-1
    #!unzip /content/detectar-placas-vehiculares-inteligencia-artificial/datasets/for_training/cars/car-2.zip -d /content/detectar-placas-vehiculares-inteligencia-artificial/datasets/for_training/cars/car-2
    !unzip /content/detectar-placas-vehiculares-inteligencia-artificial/datasets/for_training/cars/car-3.zip -d /content/detectar-placas-vehiculares-inteligencia-artificial/datasets/for_training/cars/car-3
    
    !unzip /content/detectar-placas-vehiculares-inteligencia-artificial/datasets/for_training/plates/plate-1.zip -d /content/detectar-placas-vehiculares-inteligencia-artificial/datasets/for_training/plates/plate-1
    !unzip /content/detectar-placas-vehiculares-inteligencia-artificial/datasets/for_training/plates/plate-2.zip -d /content/detectar-placas-vehiculares-inteligencia-artificial/datasets/for_training/plates/plate-2
---
    from ultralytics import YOLO
---
    #================== Configuracion base ==================
    path = '/content/detectar-placas-vehiculares-inteligencia-artificial/datasets/for_training/cars/'
    folder = 'car-1'
    epochs = 100
    output = '/models/asd'
    
    # =======================================================
    
    data =  path + folder + '/data.yaml'
    model = YOLO('datasets/models/yolov8n.pt')
    results = model.train(data=data, epochs=epochs)
    success = model.export(device=True)