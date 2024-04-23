from ultralytics import YOLO

#================== Configuracion base ==================

name = 'Vehicles-OpenImages.v1-416x416.yolov8'
category = 'cars'
epochs = 1

# =======================================================
path = '/home/kj91024/Escritorio/Proyectos/Python/Probando/datasets/for_training/' + category + '/'
data = path + name + '/data.yaml'
print('\nPath: ' + path + name)
print('yaml: ' + data+"\n")
model = YOLO('datasets/models/yolov8n.pt')
results = model.train(data=data, epochs=epochs)
#results = model.val()
#results = model('https://ultralytics.com/images/bus.jpg')
#success = model.export(format='onnx', device=True)
success = model.export(device=True)