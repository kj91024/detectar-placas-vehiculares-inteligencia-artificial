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
   
#### Configurar GPU
En Google Colab: "Runtime" > "Change runtime type", en "Hardware accelerator" es GPU, en "GPU type" es A100 y en "Runtime shape" es High-RAM sin ninguna selección.

    !nvidia-smi
    
#### Para el entrenamiento

    !pip install ultralytics

    !git clone 

    from ultralytics import YOLO
    import os
    from IPython.display import display, Image
    from IPython import display
    display.clear_output()
    !yolo checks
    

    !yolo task=detect mode=train model=yolov8s.pt conf=0.25 data=/content/drive/data
