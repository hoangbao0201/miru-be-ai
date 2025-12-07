<!-- CÀI MÔI TRƯỜNG -->
python -m venv venv
venv\Scripts\activate
deactivate

<!--  -->
pip install -r requirements.txt

python main.py


http://127.0.0.1:8000

<!-- CẤU HÌNH GPU/CPU -->
# Sử dụng GPU cho tất cả (Detection, Inpainting, OCR)
set USE_GPU=true

# Hoặc cấu hình riêng từng thành phần:
set DETECTION_DEVICE=cuda      # hoặc "cpu", "cuda", "mps" (Mac)
set INPAINT_DEVICE=cuda        # hoặc "cpu", "cuda", "mps" (Mac)
set OCR_USE_GPU=true           # hoặc "false"

# Ví dụ: Chỉ dùng GPU cho Detection và OCR, CPU cho Inpainting
set DETECTION_DEVICE=cuda
set INPAINT_DEVICE=cpu
set OCR_USE_GPU=true


<!-- TOOL HỖ TRỢ TRAIN -->
pip install labelImg==1.8.3
labelImg

<!-- XUẤT requirements.txt -->
pip freeze > requirements.txt

<!-- TRAIN -->
yolo detect train data=data.yaml model=yolov8n.pt epochs=50 imgsz=640

=======================================================================

<!-- FIX LỖI labelImg -->
E:\WORKSPACE\_PROJECT\AI\mino-translate-v3\train\venv\Lib\site-packages\labelImg\labelImg.py
bar.setValue(bar.value() + bar.singleStep() * units)
thành
bar.setValue(int(bar.value() + bar.singleStep() * units))
