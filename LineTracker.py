import cv2
import numpy as np
from skimage.morphology import skeletonize



def track_line(frame, blur=5):
   
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    if blur:
        gray = cv2.GaussianBlur(gray, (blur, blur), 0)

    # Black line on white background -> invert so the line becomes 255
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Morphological opening and closing
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    biggest_line = None
    biggest_line_area = 0
    
    for c in contours:
        c_area = cv2.contourArea(c)
        if c_area>biggest_line_area:
            biggest_line = c
            biggest_line_area = c_area

    #Skeletonize to collapse into a single 1D image
    filled = cv2.drawContours(np.zeros_like(mask), [biggest_line], -1, 255, -1) & mask
    skel = skeletonize(filled > 0).astype(np.uint8) * 255


    return skel


if __name__ == "__main__":
    frame = cv2.imread("test_image_line.png")
    cv2.imshow("original frame", frame) 

    mask = track_line(frame)  
    cv2.imshow("output", mask)  

    cv2.waitKey()
    cv2.destroyAllWindows()
