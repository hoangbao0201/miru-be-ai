"""High level orchestration for image translation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np

from ..core.config import AppSettings
from ..detection import TextBlockDetector
from ..inpainting.lama import LaMa
from ..ocr.ocr import OCRProcessor
from ..rendering.render import draw_text, get_best_render_area
from ..translator import TranslationService
from ..utils.pipeline_utils import generate_mask
from ..utils.translator_utils import format_translations
from ..utils.textblock import TextBlock


@dataclass
class TranslationContext:
    """Context returned by the pipeline after processing."""

    rendered_image: np.ndarray
    blocks: List[TextBlock]
    source_lang: str
    target_lang: str
    render_font_path: str


class ImageTranslationPipeline:
    """Co-ordinates detection, OCR, translation, inpainting and rendering."""

    def __init__(
        self,
        settings: AppSettings,
        detector: TextBlockDetector | None = None,
        ocr_processor: OCRProcessor | None = None,
        inpainter: LaMa | None = None,
        translator: TranslationService | None = None,
    ) -> None:
        self.settings = settings
        self.language_codes = settings.language_codes
        self.detector = detector or self._build_detector()
        self.ocr = ocr_processor or OCRProcessor(use_gpu=settings.ocr.use_gpu)
        self.inpainter = inpainter or LaMa(device=settings.inpainting.device)
        self.translation_service = translator or TranslationService(settings.translator)

    def _build_detector(self) -> TextBlockDetector:
        """Instantiate the YOLO based detector."""
        detection = self.settings.detection
        return TextBlockDetector(
            detection.bubble_model_path,
            detection.text_seg_model_path,
            detection.text_detect_model_path,
            detection.device,
        )

    def detect_and_ocr(
        self, image_bgr: np.ndarray, source_lng_cd: str
    ) -> List[TextBlock]:
        """
        Chỉ detect và OCR, không dịch, không inpainting.
        Trả về danh sách blocks với text_bbox.
        """
        source_lang = self.language_codes.get(source_lng_cd, "English")
        
        block_list = self.detector.detect(image_bgr)
        self.ocr.initialize(source_lang, source_lng_cd)
        block_list = self.ocr.process(image_bgr, block_list)
        
        return block_list

    def process_with_selected_boxes(
        self,
        image_bgr: np.ndarray,
        source_lng_cd: str,
        target_lng_cd: str,
        selected_boxes: List[List[int]],
    ) -> dict:
        """
        Xử lý với các box đã chọn:
        - Detect và OCR tất cả
        - Lọc chỉ giữ lại các box phù hợp với selected_boxes
        - Inpaint để xóa text
        - Dịch
        - Trả về ảnh đã xóa text (KHÔNG render text lên)
        """
        source_lang = self.language_codes.get(source_lng_cd, "English")
        target_lang = self.language_codes.get(target_lng_cd, "Vietnamese")

        # Detect và OCR tất cả
        block_list = self.detector.detect(image_bgr)
        self.ocr.initialize(source_lang, source_lng_cd)
        block_list = self.ocr.process(image_bgr, block_list)

        # Lọc blocks: chỉ giữ lại những block có text_bbox phù hợp với selected_boxes
        from ..utils.pipeline_utils import does_rectangle_fit, is_mostly_contained
        from ..detection import calculate_iou
        
        filtered_blocks = []
        for blk in block_list:
            if blk.xyxy is None:
                continue
            blk_bbox = blk.xyxy.tolist() if hasattr(blk.xyxy, 'tolist') else list(blk.xyxy)
            
            # Kiểm tra xem block này có phù hợp với bất kỳ selected_box nào không
            for selected_box in selected_boxes:
                # Tính IoU để kiểm tra overlap
                try:
                    iou = calculate_iou(selected_box, blk_bbox)
                    # Nếu IoU > 0.3 hoặc một box chứa box kia, thì coi là phù hợp
                    if iou > 0.3 or \
                       does_rectangle_fit(selected_box, blk_bbox) or \
                       does_rectangle_fit(blk_bbox, selected_box) or \
                       is_mostly_contained(selected_box, blk_bbox, 0.5) or \
                       is_mostly_contained(blk_bbox, selected_box, 0.5):
                        filtered_blocks.append(blk)
                        break
                except Exception:
                    # Fallback: kiểm tra overlap đơn giản
                    if does_rectangle_fit(selected_box, blk_bbox) or \
                       does_rectangle_fit(blk_bbox, selected_box) or \
                       is_mostly_contained(selected_box, blk_bbox, 0.5) or \
                       is_mostly_contained(blk_bbox, selected_box, 0.5):
                        filtered_blocks.append(blk)
                        break

        if not filtered_blocks:
            raise ValueError("Không tìm thấy block nào phù hợp với các box đã chọn")

        # Inpaint để xóa text
        inpaint_config = self.settings.inpainting.to_config()
        mask = generate_mask(image_bgr, filtered_blocks)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        inpainted = self.inpainter(image_rgb, mask, inpaint_config)
        inpainted = cv2.convertScaleAbs(inpainted)

        # Dịch
        filtered_blocks = self.translation_service.translate_blocks(
            source_lang, target_lang, filtered_blocks
        )

        # Format translations
        format_translations(
            filtered_blocks, target_lng_cd, upper_case=self.settings.render.upper_case
        )

        # Trả về ảnh đã xóa text (KHÔNG render text lên)
        return {
            "inpainted_image": inpainted,
            "blocks": filtered_blocks,
            "source_lang": source_lang,
            "target_lang": target_lang,
        }

    def process(
        self, image_bgr: np.ndarray, source_lng_cd: str, target_lng_cd: str
    ) -> TranslationContext:
        """Execute the full translation pipeline on a BGR image."""
        source_lang = self.language_codes.get(source_lng_cd, "English")
        target_lang = self.language_codes.get(target_lng_cd, "Vietnamese")

        block_list = self.detector.detect(image_bgr)
        self.ocr.initialize(source_lang, source_lng_cd)
        block_list = self.ocr.process(image_bgr, block_list)

        inpaint_config = self.settings.inpainting.to_config()
        mask = generate_mask(image_bgr, block_list)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        inpainted = self.inpainter(image_rgb, mask, inpaint_config)
        inpainted = cv2.convertScaleAbs(inpainted)

        block_list = self.translation_service.translate_blocks(
            source_lang, target_lang, block_list
        )

        render_settings = self.settings.render
        format_translations(
            block_list, target_lng_cd, upper_case=render_settings.upper_case
        )
        get_best_render_area(block_list, image_bgr, inpainted)

        for blk in block_list:
            blk.alignment = render_settings.alignment

        rendered = draw_text(
            inpainted,
            block_list,
            render_settings.font_path,
            colour=render_settings.color,
            init_font_size=render_settings.max_font_size,
            min_font_size=render_settings.min_font_size,
            outline=render_settings.outline,
        )

        return TranslationContext(
            rendered_image=rendered,
            blocks=block_list,
            source_lang=source_lang,
            target_lang=target_lang,
            render_font_path=render_settings.font_path,
        )

