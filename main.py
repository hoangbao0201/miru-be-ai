from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import cv2
import numpy as np
from io import BytesIO

from modules.core import load_settings
from modules.pipeline import ImageTranslationPipeline
from modules.rendering.render import draw_text
from modules.utils.serialization import serialize_text_blocks, deserialize_text_blocks


app = Flask(__name__)
CORS(app)

settings = load_settings()

# Pre-initialize EasyOCR để tránh delay khi xử lý
try:
    import easyocr
    _easyocr_reader = easyocr.Reader(['en'], gpu=settings.ocr.use_gpu)
except Exception:
    _easyocr_reader = None

# Pre-initialize OCR processor với EasyOCR reader đã khởi tạo sẵn
from modules.ocr.ocr import OCRProcessor
ocr_processor = OCRProcessor(easyocr_reader=_easyocr_reader, use_gpu=settings.ocr.use_gpu)
pipeline = ImageTranslationPipeline(settings=settings, ocr_processor=ocr_processor)


def _resize_image_if_large(image_bgr: np.ndarray, max_dimension: int = 2048) -> tuple[np.ndarray, float]:
    """
    Resize ảnh nếu quá lớn để tăng tốc xử lý.
    
    Args:
        image_bgr: Ảnh BGR
        max_dimension: Kích thước tối đa (chiều dài hoặc rộng)
    
    Returns:
        Tuple (resized_image, scale_factor)
    """
    h, w = image_bgr.shape[:2]
    max_size = max(h, w)
    
    if max_size <= max_dimension:
        return image_bgr, 1.0
    
    scale = max_dimension / max_size
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale


def _decode_image_from_request() -> tuple[np.ndarray, str, str]:
    """
    Helper function để decode ảnh và lấy tham số từ request.
    Tối ưu: chỉ đọc file một lần.
    
    Returns:
        Tuple (image_bgr, source_lng_cd, target_lng_cd)
    """
    if "image" not in request.files:
        raise ValueError("No image file uploaded")
    
    image_file = request.files["image"]
    source_lng_cd = request.form.get("source_lng_cd", "en")
    target_lng_cd = request.form.get("target_lng_cd", "vi")
    
    # Tối ưu: đọc file một lần
    file_bytes = image_file.read()
    file_array = np.frombuffer(file_bytes, dtype=np.uint8)
    image_bgr = cv2.imdecode(file_array, cv2.IMREAD_COLOR)
    
    if image_bgr is None:
        raise ValueError("Invalid image file")
    
    # Resize nếu ảnh quá lớn để tăng tốc xử lý
    image_bgr, _ = _resize_image_if_large(image_bgr, max_dimension=2048)
    
    return image_bgr, source_lng_cd, target_lng_cd


@app.route("/api/translate/images", methods=["POST"])
def upload_image():
    """
    API 1: Gửi ảnh, trả về ảnh đã dịch.
    
    Request:
        - image: File ảnh
        - source_lng_cd: Mã ngôn ngữ nguồn (mặc định: "en")
        - target_lng_cd: Mã ngôn ngữ đích (mặc định: "vi")
    
    Returns:
        File ảnh JPEG đã dịch
    """
    try:
        image_bgr, source_lng_cd, target_lng_cd = _decode_image_from_request()
        context = pipeline.process(image_bgr, source_lng_cd, target_lng_cd)
        rendered_bgr = cv2.cvtColor(context.rendered_image, cv2.COLOR_RGB2BGR)

        # Tối ưu JPEG encoding: cân bằng quality và tốc độ
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, 85]  # Giảm từ 95 xuống 85 để nhanh hơn
        _, img_encoded = cv2.imencode(".jpg", rendered_bgr, encode_params)
        return send_file(
            BytesIO(img_encoded.tobytes()),
            mimetype="image/jpeg",
            as_attachment=True,
            download_name="processed_image.jpg",
        )
    except Exception as err:
        return jsonify({"error": str(err)}), 500


@app.route("/api/translate/images/select/text", methods=["POST"])
def upload_image_select_text():
    """
    API: Detect và OCR, trả về tọa độ các text boxes.
    
    Request:
        - image: File ảnh
        - source_lng_cd: Mã ngôn ngữ nguồn (mặc định: "en")
    
    Returns:
        JSON chứa:
        - boxes: Danh sách các text box với text_bbox [x1, y1, x2, y2]
    """
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image file uploaded"}), 400
        
        image_file = request.files["image"]
        source_lng_cd = request.form.get("source_lng_cd", "en")
        
        # Tối ưu: đọc file một lần
        file_bytes = image_file.read()
        file_array = np.frombuffer(file_bytes, dtype=np.uint8)
        image_bgr = cv2.imdecode(file_array, cv2.IMREAD_COLOR)
        
        if image_bgr is None:
            return jsonify({"error": "Invalid image file"}), 400
        
        # Resize nếu ảnh quá lớn
        image_bgr, _ = _resize_image_if_large(image_bgr, max_dimension=2048)
        
        # Chỉ detect và OCR
        block_list = pipeline.detect_and_ocr(image_bgr, source_lng_cd)
        
        # Serialize chỉ text_bbox - sử dụng hàm serialize_text_blocks để đảm bảo format giống nhau
        from modules.utils.serialization import _bbox_to_list
        
        boxes_data = []
        for idx, blk in enumerate(block_list):
            if blk.xyxy is not None:
                # Sử dụng _bbox_to_list để đảm bảo format giống với API detail
                bbox = _bbox_to_list(blk.xyxy)
                if bbox is not None:
                    boxes_data.append({
                        "id": idx,
                        "text_bbox": bbox,
                    })
        
        return jsonify({"boxes": boxes_data})
    except Exception as err:
        return jsonify({"error": str(err)}), 500


@app.route("/api/translate/images/detail", methods=["POST"])
def upload_image_detail():
    """
    API 2: Gửi ảnh với selected_boxes, trả về ảnh đã xóa text và thông tin chi tiết.
    
    Request:
        - image: File ảnh
        - source_lng_cd: Mã ngôn ngữ nguồn (mặc định: "en")
        - target_lng_cd: Mã ngôn ngữ đích (mặc định: "vi")
        - selected_boxes: JSON string danh sách boxes [[x1, y1, x2, y2], ...]
    
    Returns:
        JSON chứa:
        - inpainted_image: Base64 encoded ảnh đã xóa text
        - blocks: Danh sách các text block với:
            - text: Text nhận dạng được (OCR)
            - translation: Text đã dịch
            - text_bbox: Vị trí text nhận dạng [x1, y1, x2, y2]
            - bubble_bbox: Vị trí bubble (nếu có) [x1, y1, x2, y2]
            - inpaint_bboxes: Danh sách vị trí xóa text [[x1, y1, x2, y2], ...]
            - text_class: Loại text (text_bubble, text_free)
        - source_lang: Tên ngôn ngữ nguồn
        - target_lang: Tên ngôn ngữ đích
        - source_lng_cd: Mã ngôn ngữ nguồn
        - target_lng_cd: Mã ngôn ngữ đích
    """
    import json
    import base64
    
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image file uploaded"}), 400
        
        image_file = request.files["image"]
        source_lng_cd = request.form.get("source_lng_cd", "en")
        target_lng_cd = request.form.get("target_lng_cd", "vi")
        selected_boxes_json = request.form.get("selected_boxes", "[]")
        
        # Parse selected_boxes
        try:
            selected_boxes = json.loads(selected_boxes_json)
        except json.JSONDecodeError:
            return jsonify({"error": "Invalid selected_boxes format"}), 400
        
        # Tối ưu: đọc file một lần
        file_bytes = image_file.read()
        file_array = np.frombuffer(file_bytes, dtype=np.uint8)
        image_bgr = cv2.imdecode(file_array, cv2.IMREAD_COLOR)
        
        if image_bgr is None:
            return jsonify({"error": "Invalid image file"}), 400
        
        # Resize nếu ảnh quá lớn
        image_bgr, _ = _resize_image_if_large(image_bgr, max_dimension=2048)
        
        # Xử lý với selected boxes
        result = pipeline.process_with_selected_boxes(
            image_bgr, source_lng_cd, target_lng_cd, selected_boxes
        )
        
        # Encode ảnh đã xóa text thành base64 (ảnh đang ở định dạng BGR từ OpenCV)
        _, img_encoded = cv2.imencode('.jpg', result["inpainted_image"])
        img_base64 = base64.b64encode(img_encoded.tobytes()).decode('utf-8')
        
        blocks_data = serialize_text_blocks(result["blocks"])
        
        return jsonify({
            "inpainted_image": f"data:image/jpeg;base64,{img_base64}",
            "blocks": blocks_data,
            "source_lang": result["source_lang"],
            "target_lang": result["target_lang"],
            "source_lng_cd": source_lng_cd,
            "target_lng_cd": target_lng_cd,
        })
    except Exception as err:
        return jsonify({"error": str(err)}), 500


@app.route("/api/translate/images/render", methods=["POST"])
def render_translated_text():
    """
    API 3: Nhận ảnh đã xóa text + danh sách blocks (đã chỉnh sửa), trả về ảnh đã chèn text.

    Request:
        - image: File ảnh đã xóa text
        - blocks: JSON string danh sách block với translation và text_bbox
    """
    import json
    import base64

    try:
        if "image" not in request.files:
            return jsonify({"error": "No image file uploaded"}), 400

        blocks_json = request.form.get("blocks")
        if not blocks_json:
            return jsonify({"error": "No blocks payload provided"}), 400

        try:
            blocks_payload = json.loads(blocks_json)
        except json.JSONDecodeError:
            return jsonify({"error": "Invalid blocks format"}), 400

        if not isinstance(blocks_payload, list) or len(blocks_payload) == 0:
            return jsonify({"error": "Blocks payload must be a non-empty list"}), 400

        image_file = request.files["image"]
        # Tối ưu: đọc file một lần
        file_bytes = image_file.read()
        file_array = np.frombuffer(file_bytes, dtype=np.uint8)
        clean_bgr = cv2.imdecode(file_array, cv2.IMREAD_COLOR)

        if clean_bgr is None:
            return jsonify({"error": "Invalid image file"}), 400

        text_blocks = deserialize_text_blocks(blocks_payload)
        if not text_blocks:
            return jsonify({"error": "No valid blocks to render"}), 400

        render_settings = settings.render
        default_line_spacing = getattr(render_settings, "line_spacing", 1)
        for blk in text_blocks:
            # Bổ sung các giá trị mặc định nếu FE không gửi
            if not blk.alignment:
                blk.alignment = render_settings.alignment
            if not blk.font_color:
                blk.font_color = render_settings.color
            if blk.min_font_size <= 0:
                blk.min_font_size = render_settings.min_font_size
            if blk.max_font_size <= 0:
                blk.max_font_size = render_settings.max_font_size
            if blk.line_spacing <= 0:
                blk.line_spacing = default_line_spacing

        rendered_rgb = draw_text(
            clean_bgr,
            text_blocks,
            render_settings.font_path,
            colour=render_settings.color,
            init_font_size=render_settings.max_font_size,
            min_font_size=render_settings.min_font_size,
            outline=render_settings.outline,
        )
        rendered_bgr = cv2.cvtColor(rendered_rgb, cv2.COLOR_RGB2BGR)
        _, img_encoded = cv2.imencode(".jpg", rendered_bgr)
        img_base64 = base64.b64encode(img_encoded.tobytes()).decode("utf-8")

        return jsonify({
            "rendered_image": f"data:image/jpeg;base64,{img_base64}",
        })
    except Exception as err:
        return jsonify({"error": str(err)}), 500


if __name__ == "__main__":
    app.run(debug=True)