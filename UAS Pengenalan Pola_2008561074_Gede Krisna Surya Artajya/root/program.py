from ultralytics import YOLO

img_path = input("Input nama gambar yang ingin dideteksi: ")
model = YOLO('./runs/classify/train10/weights/best.pt')  # load a custom model

# Predict with the model
results = model(img_path, save=True)