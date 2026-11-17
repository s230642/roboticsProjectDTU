import os

os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

import cv2
import numpy as np
from skimage.morphology import skeletonize



def track_line(frame, blur=5, max_threshold=120, min_thinness=100, max_line_width=21):
   
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    if blur:
        gray = cv2.GaussianBlur(gray, (blur, blur), 0)

    # Black line on white background -> invert so the line becomes 255.
    # Otsu picks the threshold, but never above max_threshold, so light gray background is not counted as line.
    otsu, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, mask = cv2.threshold(gray, min(otsu, max_threshold), 255, cv2.THRESH_BINARY_INV)
        
    # Close small gaps in the line. No opening: it cuts the line at thin spots, and specks are ignored below anyway
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    # Remove dark areas wider than the line (shadows, dark corners), so they cannot merge with the line.
    # An opening wider than the line erases the line but keeps the wide areas; subtract those (slightly grown).
    wide = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((max_line_width, max_line_width), np.uint8))
    mask = mask & ~cv2.dilate(wide, np.ones((5, 5), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Keep only long, thin shapes (perimeter^2 / area: circle ~13, line >> 100) and pick the longest
    lines = [c for c in contours if cv2.arcLength(c, True) ** 2 / max(cv2.contourArea(c), 1) > min_thinness]
    if not lines:
        return np.zeros_like(mask)
    biggest_line = max(lines, key=lambda c: cv2.arcLength(c, True))

    #Skeletonize to collapse into a single 1D image
    filled = cv2.drawContours(np.zeros_like(mask), [biggest_line], -1, 255, -1) & mask
    skel = skeletonize(filled > 0).astype(np.uint8) * 255


    return skel


if __name__ == "__main__":
    cam = cv2.VideoCapture(0)
    ret, frame = cam.read()

    cv2.imshow("original frame", frame) 

    mask = track_line(frame)  
    cv2.imshow("output", mask)  

    cv2.waitKey()
    cv2.destroyAllWindows()
    cam.release()
