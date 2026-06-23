import os
import gc
import torch
from flask import Flask, request, jsonify
from ultralytics import YOLO
from PIL import Image

app = Flask(__name__)

# Khống chế chặt chẽ tài nguyên hệ thống
torch.set_num_threads(1)

# Tải model toàn cục
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'best.pt')
model = YOLO(MODEL_PATH)

@app.route('/predict', methods=['POST'])
def predict():
    # SỬA TÊN KEY THÀNH 'filecccdt' ĐỂ KHỚP VỚI RAILS GỬI LÊN
    if 'filecccdt' not in request.files:
        return jsonify({"success": False, "message": "Không tìm thấy key 'filecccdt' trong dữ liệu gửi lên"}), 400
        
    file = request.files['filecccdt']
    if file.filename == '':
        return jsonify({"success": False, "message": "Tên file rỗng"}), 400

    try:
        # Đọc ảnh trực tiếp từ stream để tránh ghi file xuống ổ đĩa gây tốn tài nguyên
        img = Image.open(file.stream).convert('RGB')
        
        # Cấu hình giảm tải RAM tối đa khi Predict
        results = model(
            img, 
            imgsz=256,      # Giảm kích thước ảnh xuống 256 để chạy nhẹ hơn nữa
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
        
        # Ép giải phóng bộ nhớ triệt để ngay khi có kết quả
        del img
        del results
        gc.collect()
        
        return jsonify(output_data)

    except Exception as e:
        # Trả về chi tiết lỗi cụ thể thay vì trả trang HTML 500 mặc định
        return jsonify({"success": False, "message": f"Lỗi xử lý tại Flask: {str(e)}"}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)