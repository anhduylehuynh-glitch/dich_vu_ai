import os
import gc
import torch
from flask import Flask, request, jsonify
from ultralytics import YOLO
from PIL import Image

app = Flask(__name__)

# Khống chế chặt chẽ tài nguyên hệ thống (Tránh sinh đa tiến trình tốn RAM)
torch.set_num_threads(1)

# Tải model toàn cục
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'best.pt')
model = YOLO(MODEL_PATH)

@app.route('/predict', methods=['POST'])
def predict():
    if 'filecccdt' not in request.files:
        return jsonify({"success": False, "message": "Không tìm thấy key 'filecccdt' trong dữ liệu gửi lên"}), 400
        
    file = request.files['filecccdt']
    if file.filename == '':
        return jsonify({"success": False, "message": "Tên file rỗng"}), 400

    try:
        # Đọc ảnh trực tiếp từ bộ nhớ đệm
        img = Image.open(file.stream)
        
        # Chuyển đổi sang hệ màu RGB nếu ảnh tải lên là dạng RGBA (tránh lỗi kênh Alpha)
        if img.mode == 'RGBA':
            img = img.convert('RGB')
            
        # Ép chuẩn ma trận 640x640 giống hệt như file detect.py để đảm bảo độ chính xác
        img_resized = img.resize((640, 640))
        
        # Chạy dự đoán với độ phân giải chuẩn của model tuyển sinh
        results = model(
            img_resized, 
            imgsz=640,      # BẮT BUỘC ĐỂ 640 để YOLO nhận diện được chữ nhỏ trên CCCD
            conf=0.35, 
            verbose=False, 
            device='cpu'
        )
        
        detections = []
        for result in results:
            if hasattr(result, 'boxes') and result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    cls_name = model.names[cls_id]

                    detections.append({
                        "class": cls_name,
                        "confidence": conf,
                        "box": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)]
                    })
        
        output_data = {
            "success": True,
            "count": len(detections),
            "detections": detections
        }
        
        # Thu dọn rác và giải phóng bộ nhớ đệm ngay lập tức
        del img
        del img_resized
        del results
        gc.collect()
        
        return jsonify(output_data)

    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)