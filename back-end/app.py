import os
import torch
from torchvision import transforms
from torchvision.models import resnet34
from PIL import Image
from flask import Flask, request, jsonify

# Tạo Flask app
app = Flask(__name__)

# Tải mô hình đã huấn luyện
class CustomResNet34(torch.nn.Module):
    def __init__(self, output_classes):
        super(CustomResNet34, self).__init__()
        self.base_model = resnet34(weights=None)
        in_features = self.base_model.fc.in_features
        self.base_model.fc = torch.nn.Linear(in_features, output_classes)

    def forward(self, x):
        return self.base_model(x)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CustomResNet34(output_classes=2).to(device)
model.load_state_dict(torch.load("DogandCat_resnet34.pth", map_location=device))
model.eval()

# Chuẩn bị transform cho ảnh
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Hàm xử lý ảnh và dự đoán
def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    return predicted.item()  # Trả về 0 hoặc 1

# API chính: upload ảnh và trả về kết quả
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded!"}), 400
    
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected!"}), 400

    file_path = os.path.join("uploads", file.filename)
    file.save(file_path)

    # Dự đoán nhãn
    label = predict_image(file_path)
    os.remove(file_path)  # Xóa file sau khi xử lý
    
    label_str = "cat" if label == 0 else "dog"
    return jsonify({"label": label_str})  # Trả về nhãn dạng chuỗi

# Chạy ứng dụng Flask
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
