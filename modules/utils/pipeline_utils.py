import cv2
import numpy as np
from typing import List

from .textblock import TextBlock, sort_textblock_rectangles
from ..detection import is_mostly_contained


def generate_mask(
    img: np.ndarray, blk_list: List[TextBlock], default_padding: int = 5
) -> np.ndarray:
    """Build mask that covers all inpainting regions."""
    h, w, c = img.shape
    mask = np.zeros((h, w), dtype=np.uint8)  # Start with a black mask
    
    for blk in blk_list:
        bboxes = blk.inpaint_bboxes
        if bboxes is None or len(bboxes) == 0:
            continue
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            
            # Determine kernel size for dilation
            kernel_size = default_padding
            if hasattr(blk, 'source_lang') and blk.source_lang not in ['ja', 'ko']:
                kernel_size = 3
            if hasattr(blk, 'text_class') and blk.text_class == 'text_bubble':
                # Calculate the minimal distance from the mask to the bounding box edges
                min_distance_to_bbox = min(
                    x1 - blk.bubble_xyxy[0],  # left side
                    blk.bubble_xyxy[2] - x2,  # right side
                    y1 - blk.bubble_xyxy[1],  # top side
                    blk.bubble_xyxy[3] - y2   # bottom side
                )
                # Adjust kernel size if necessary
                if kernel_size >= min_distance_to_bbox:
                    kernel_size = max(1, int(min_distance_to_bbox - (0.2 * min_distance_to_bbox)))
            
            # Create a temporary mask for this bbox
            temp_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.rectangle(temp_mask, (x1, y1), (x2, y2), 255, -1)
            
            # Create kernel for dilation
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            
            # Dilate the temporary mask
            dilated_mask = cv2.dilate(temp_mask, kernel, iterations=4)
            
            # Add the dilated mask to the main mask
            mask = cv2.bitwise_or(mask, dilated_mask)
    
    return mask


def does_rectangle_fit(bigger_rect, smaller_rect):
    """Check whether a rectangle fits entirely inside another."""
    x1, y1, x2, y2 = bigger_rect
    px1, py1, px2, py2 = smaller_rect
    
    # Ensure the coordinates are properly ordered
    # first rectangle
    left1, top1, right1, bottom1 = min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)
    # second rectangle
    left2, top2, right2, bottom2 = min(px1, px2), min(py1, py2), max(px1, px2), max(py1, py2)
    
    # Check if the second rectangle fits within the first
    fits_horizontally = left1 <= left2 and right1 >= right2
    fits_vertically = top1 <= top2 and bottom1 >= bottom2
    
    return fits_horizontally and fits_vertically


def lists_to_blk_list(
    blk_list: List[TextBlock], texts_bboxes: List, texts_string: List
):
    """Assign recognized text strings back to text blocks."""
    group = list(zip(texts_bboxes, texts_string))  

    for blk in blk_list:
        blk_entries = []
        
        for line, text in group:
            if blk.bubble_xyxy is not None:
                if does_rectangle_fit(blk.bubble_xyxy, line):
                    blk_entries.append((line, text))  
                elif is_mostly_contained(blk.bubble_xyxy, line, 0.5):
                    blk_entries.append((line, text)) 

            elif does_rectangle_fit(blk.xyxy, line):
                blk_entries.append((line, text)) 
            elif is_mostly_contained(blk.xyxy, line, 0.5):
                blk_entries.append((line, text)) 

        # Sort and join text entries
        sorted_entries = sort_textblock_rectangles(blk_entries, blk.source_lang_direction)
        if blk.source_lang in ['ja', 'zh']:
            blk.text = ''.join(text for bbox, text in sorted_entries)
        else:
            blk.text = ' '.join(text for bbox, text in sorted_entries)

    return blk_list
