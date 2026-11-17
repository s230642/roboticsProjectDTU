"""Check that track_line finds a black line on a white and light gray background, and ignores the gray."""
import cv2
import numpy as np

from LineTracker import track_line


def make_scene(gray_level=170, line_level=30):
    """White image with a large light gray area and a curvy black line crossing both."""
    img = np.full((480, 640, 3), 235, np.uint8)
    img[:, 320:] = gray_level
    pts = np.array([[60 + 520 * t, 240 + 120 * np.sin(6 * t)] for t in np.linspace(0, 1, 300)], np.int32)
    cv2.polylines(img, [pts], False, (line_level,) * 3, 6)
    line = np.zeros(img.shape[:2], np.uint8)
    cv2.polylines(line, [pts], False, 1, 6)
    return img, line


if __name__ == "__main__":
    cv2.imshow = lambda *args: None  # track_line shows debug windows; not needed here
    for max_threshold in (255, 120):  # 255 = plain Otsu, 120 = default
        for gray_level in (140, 170, 200, 235):  # 235 = no gray area, only white
            img, line = make_scene(gray_level)
            img = cv2.add(img, np.random.default_rng(0).integers(0, 12, img.shape, dtype=np.uint8))  # sensor noise
            skel = track_line(img, max_threshold=max_threshold) > 0
            on_line = (skel & (line > 0)).sum() / skel.sum() if skel.any() else 0
            span = np.ptp(np.nonzero(skel)[1]) if skel.any() else 0
            print(f"max_threshold {max_threshold}, gray background {gray_level}: "
                  f"{on_line:.0%} of skeleton on the line, spans {span} of 520 px")

    # Real camera photo: blurry gray line, darker corners that are bigger in area than the line
    skel = track_line(cv2.imread("test_image_dark_corners.png")) > 0
    ys, xs = np.nonzero(skel)
    print(f"dark-corners photo: skeleton spans x {xs.min()}-{xs.max()} (line runs ~0-550), "
          f"{skel[:70, 430:].sum()} px in the dark top-right corner (should be 0)")

    # Real camera photo: line has a thin spot near the bottom-left end, which morphological opening used to cut off
    skel = track_line(cv2.imread("test_image_thin_spot.png")) > 0
    xs = np.nonzero(skel)[1]
    print(f"thin-spot photo: skeleton spans x {xs.min()}-{xs.max()} (line runs ~60-595)")

    # Same photo with the top-right corner darkened until it touches the end of the line
    img = cv2.imread("test_image_thin_spot.png")
    yy, xx = np.mgrid[:480, :640]
    shade = np.clip((250 - np.hypot(xx - 640, yy)) / 150, 0, 1)[..., None]  # dark within ~190 px of the corner
    img = (img * (1 - 0.5 * shade)).astype(np.uint8)
    skel = track_line(img) > 0
    xs = np.nonzero(skel)[1]
    print(f"dark corner touching the line: skeleton spans x {xs.min()}-{xs.max()} (line runs ~60-595), "
          f"{skel[:120, 600:].sum()} px in the corner beyond the line end (should be 0)")
