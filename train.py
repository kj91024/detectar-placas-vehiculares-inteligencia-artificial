from ultralytics import YOLO

#================== Configuracion base ==================
name = 'vehicle-and-license-plate.v3i.yolov8'
epochs = 3
output = '/models/vehicle-and-license-plate.v3i.yolov8'

# =======================================================

data = 'datasets/for_training/cars/' + name + '/data.yaml'
model = YOLO('datasets/models/yolov8n.pt')
results = model.train(data=data, epochs=epochs, save_dir="runs/detect/train")
#results = model.val()
#results = model('https://ultralytics.com/images/bus.jpg')
success = model.export(format='onnx', device=True)
#success = model.export(format='onnx', device=0)