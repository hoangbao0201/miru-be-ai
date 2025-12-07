import cv2
import os
import easyocr
import numpy as np
from typing import List
from paddleocr import PaddleOCR
from ..utils.pipeline_utils import lists_to_blk_list
from ..utils.textblock import TextBlock, adjust_text_line_coordinates

language_codes = {
    "Chinese": "zh",
    "English": "en",
    "Vietnamese": "vi",
}

class OCRProcessor:
    def __init__(self, easyocr_reader=None, use_gpu=False):
        self.manga_ocr = None
        self.easy_ocr = easyocr_reader  # Cho phép truyền reader đã khởi tạo sẵn
        self.use_gpu = use_gpu  # Lưu cấu hình GPU để dùng khi khởi tạo lazy

    def initialize(self, source_lang: str, source_lng_cd: str):
        self.source_lang = source_lang
        self.source_lng_cd = source_lng_cd

    def set_source_orientation(self, blk_list: List[TextBlock]):
        source_lang_code = self.source_lng_cd
        for blk in blk_list:
            blk.source_lang = source_lang_code

    def process(self, img: np.ndarray, blk_list: List[TextBlock]):
        self.set_source_orientation(blk_list)
        if self.source_lang == 'Chinese':
            return self._ocr_paddle(img, blk_list)
        return self._ocr_default(img, blk_list, self.source_lang)
    
    def _ocr_default(self, img: np.ndarray, blk_list: List[TextBlock], source_language: str, device: str = 'cpu', expansion_percentage: int = 5):
        for blk in blk_list:
            if blk.bubble_xyxy is not None:
                x1, y1, x2, y2 = blk.bubble_xyxy
            else:
                x1, y1, x2, y2 = adjust_text_line_coordinates(blk.xyxy, expansion_percentage, expansion_percentage, img)

            if x1 < x2 and y1 < y2:
                if source_language == 'English':
                    if self.easy_ocr is None:
                        self.easy_ocr = easyocr.Reader(['en'], gpu=self.use_gpu)

                    result = self.easy_ocr.readtext(img[y1:y2, x1:x2], paragraph=True)
                    texts = []
                    for r in result:
                        if r is None:
                            continue
                        texts.append(r[1])
                    text = ' '.join(texts)
                    blk.text = text

                    print("text: ", text)

            else:
                print('Invalid textbbox to target img')
                blk.text = ['']

        return blk_list
    
    def _ocr_paddle(self, img: np.ndarray, blk_list: List[TextBlock]):
        
        ch_ocr = PaddleOCR(lang='ch', show_log=False)
        result = ch_ocr.ocr(img)
        result = result[0]

        texts_bboxes = [tuple(coord for point in bbox for coord in point) for bbox, _ in result] if result else []
        condensed_texts_bboxes = [(x1, y1, x2, y2) for (x1, y1, x2, y1_, x2_, y2, x1_, y2_) in texts_bboxes]

        texts_string = [line[1][0] for line in result] if result else []

        blk_list = lists_to_blk_list(blk_list, condensed_texts_bboxes, texts_string)

        return blk_list