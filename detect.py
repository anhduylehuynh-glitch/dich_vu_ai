import os
import sys
import json
import time

# Ép lưu cache cấu hình vào thư mục /tmp được cấp quyền trên Render
os.environ["YOLO_CONFIG_DIR"] = "/tmp/Ultralytics"
os.environ["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
os.environ["PYTHONPYCACHEPREFIX"] = "/tmp/pycache"
os.environ["OMP_NUM_THREADS"] = "1"

from ultralytics import YOLO
from PIL import Image

try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(BASE_DIR, "best.pt")
    model = YOLO(model_path)

    image_path = sys.argv[1]
    img = Image.open(image_path)

    # Chuyển đổi hệ màu RGBA sang RGB để tránh lỗi kênh Alpha
    if img.mode == "RGBA":
        img = img.convert("RGB")

    # Ép chuẩn ma trận vuông 640x640 khớp chính xác với tập dữ liệu train
    img_resized = img.resize((640, 640))

    tmp_path = image_path + "_predict.jpg"
    img_resized.save(tmp_path, "JPEG")

    # Thực hiện nhận diện với độ phân giải chuẩn và ngưỡng tin cậy 0.35 nhạy bén
    results = model(tmp_path, imgsz=640, conf=0.35, verbose=False)
    boxes = results[0].boxes

    if len(boxes) > 0:
        best_box = boxes[0]
        label_name = model.names[int(best_box.cls[0])]
        print(json.dumps({"success": True, "label": label_name}))
    else:
        print(json.dumps({"success": False, "message": "Không nhận diện được đặc trưng thẻ"}))

    # Dọn dẹp file tạm
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

except Exception as e:
    print(json.dumps({"success": False, "message": str(e)}))