"""Utilities for serializing TextBlock objects to JSON-compatible formats."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from .textblock import TextBlock


def _bbox_to_list(bbox: np.ndarray | tuple | None) -> List[int] | None:
    """Chuyển đổi bounding box (numpy array hoặc tuple) thành list int."""
    if bbox is None:
        return None
    if isinstance(bbox, np.ndarray):
        # Convert numpy array to list of ints để đảm bảo format nhất quán
        return [int(x) for x in bbox.tolist()]
    if isinstance(bbox, (list, tuple)):
        return [int(x) for x in bbox]
    return None


def serialize_text_block(blk: TextBlock) -> Dict[str, Any]:
    """
    Chuyển đổi TextBlock thành dictionary có thể serialize thành JSON.
    
    Args:
        blk: TextBlock cần serialize
        
    Returns:
        Dictionary chứa:
        - text: Text nhận dạng được (OCR)
        - translation: Text đã dịch
        - text_bbox: Vị trí text nhận dạng [x1, y1, x2, y2]
        - bubble_bbox: Vị trí bubble (nếu có) [x1, y1, x2, y2]
        - inpaint_bboxes: Danh sách vị trí xóa text [[x1, y1, x2, y2], ...]
        - text_class: Loại text (text_bubble, text_free)
    """
    # Xử lý inpaint_bboxes: có thể là numpy array 2D hoặc list
    inpaint_list = []
    if blk.inpaint_bboxes is not None:
        if isinstance(blk.inpaint_bboxes, np.ndarray):
            # Nếu là numpy array 2D, iterate qua từng hàng
            if blk.inpaint_bboxes.ndim == 2:
                for i in range(len(blk.inpaint_bboxes)):
                    bbox = blk.inpaint_bboxes[i]
                    inpaint_list.append(_bbox_to_list(bbox))
            else:
                # Nếu là 1D, xử lý như một bbox duy nhất
                inpaint_list.append(_bbox_to_list(blk.inpaint_bboxes))
        elif isinstance(blk.inpaint_bboxes, (list, tuple)):
            # Nếu là list/tuple, iterate qua từng phần tử
            for bbox in blk.inpaint_bboxes:
                inpaint_list.append(_bbox_to_list(bbox))
    
    return {
        "text": blk.text or "",
        "translation": blk.translation or "",
        "text_bbox": _bbox_to_list(blk.xyxy),
        "bubble_bbox": _bbox_to_list(blk.bubble_xyxy),
        "inpaint_bboxes": inpaint_list,
        "text_class": blk.text_class or "",
    }


def serialize_text_blocks(blk_list: List[TextBlock]) -> List[Dict[str, Any]]:
    """
    Chuyển đổi danh sách TextBlock thành list dictionaries.
    
    Args:
        blk_list: Danh sách TextBlock
        
    Returns:
        List các dictionary đã serialize
    """
    return [serialize_text_block(blk) for blk in blk_list]


def deserialize_text_block(data: Dict[str, Any]) -> TextBlock | None:
    """
    Khôi phục TextBlock từ dữ liệu JSON gửi từ FE.
    """
    text_bbox = data.get("text_bbox")
    if text_bbox is None:
        return None

    bubble_bbox = data.get("bubble_bbox")
    inpaint_bboxes = data.get("inpaint_bboxes")

    return TextBlock(
        text_bbox=np.array(text_bbox, dtype=np.int32),
        bubble_bbox=np.array(bubble_bbox, dtype=np.int32) if bubble_bbox else None,
        inpaint_bboxes=inpaint_bboxes,
        text_class=data.get("text_class", ""),
        text=data.get("text", ""),
        translation=data.get("translation", ""),
        line_spacing=data.get("line_spacing", 1),
        alignment=data.get("alignment", ""),
        source_lang=data.get("source_lang", ""),
        target_lang=data.get("target_lang", ""),
        min_font_size=data.get("min_font_size", 0),
        max_font_size=data.get("max_font_size", 0),
        font_color=data.get("font_color", ""),
    )


def deserialize_text_blocks(data_list: List[Dict[str, Any]]) -> List[TextBlock]:
    """
    Chuyển list dictionary thành TextBlock list, bỏ qua các entry thiếu bbox.
    """
    blocks: List[TextBlock] = []
    for data in data_list:
        blk = deserialize_text_block(data)
        if blk is not None:
            blocks.append(blk)
    return blocks
