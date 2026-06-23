import os
import gc
import torch
from flask import Flask, request, jsonify
from ultralytics import YOLO
from PIL import Image

app = Flask(__name__)

# Khống chế số thread để không bị nổ RAM trên Render Free
torch.set_num_threads(1)

# Tải model toàn cục - chỉ chạy duy nhất 1 lần khi start server
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'best.pt')
model = YOLO(MODEL_PATH)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"success": False, "message": "Không tìm thấy file ảnh gửi lên"}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({"success": False, "message": "Tên file rỗng"}), 400

    try:
        # Đọc ảnh trực tiếp từ stream của request
        img = Image.open(file.stream).convert('RGB')
        
        # Dự đoán với kích thước ảnh nhỏ (320) để chạy cực nhẹ trên CPU Render
        results = model(img, imgsz=320, conf=0.35, verbose=False, device='cpu')
        
        detections = []
        # Duyệt qua các kết quả nhận diện được từ YOLO
        for result in results:
            for box in result.boxes:
                # Lấy tọa độ, class id và độ tự tin (confidence)
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = model.names[cls_id] # Tên class (Ví dụ: id, name, birth,...)

                detections.append({
                    "class": cls_name,
                    "confidence": conf,
                    "box": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)]
                })
        
        # Chuẩn bị dữ liệu trả về cho Rails
        output_data = {
            "success": True,
            "count": len(detections),
            "detections": detections
        }
        
        # Giải phóng bộ nhớ ngay lập tức
        del img
        del results
        gc.collect()
        
        return jsonify(output_data)

    except Exception as e:
        return jsonify({"success": False, "message": f"Lỗi xử lý AI: {str(e)}"}), 500

if __name__ == '__main__':
    # Chạy local để test nếu cần, khi lên Render nó sẽ tự dùng Gunicorn
    app.run(host='0.0.0.0', port=5000)