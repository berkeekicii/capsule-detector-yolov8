"""Run the capsule detector on a single image and show the result in a window.

Usage:
    python app/app.py [IMAGE] [WEIGHTS]

Defaults: IMAGE = test/foto1.jpg, WEIGHTS = weights/best.pt
"""
import sys

import cv2
from ultralytics import YOLO


def letterbox_no_zoom(im, new_shape=(640, 640), color=(0, 0, 0)):
    h, w = im.shape[:2]

    # only shrink, never enlarge
    r = min(new_shape[0] / h, new_shape[1] / w)
    r = min(r, 1.0)

    new_w, new_h = int(w * r), int(h * r)
    im_resized = cv2.resize(im, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # padding
    pad_w = new_shape[1] - new_w
    pad_h = new_shape[0] - new_h

    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    return cv2.copyMakeBorder(
        im_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )


def main():
    image_path = sys.argv[1] if len(sys.argv) > 1 else "test/foto1.jpg"
    weights_path = sys.argv[2] if len(sys.argv) > 2 else "weights/best.pt"

    model = YOLO(weights_path)

    img0 = cv2.imread(image_path)
    if img0 is None:
        raise SystemExit(f"Could not read image: {image_path}")

    img = letterbox_no_zoom(img0, (640, 640))
    results = model(img)

    out = results[0].plot()
    cv2.imshow("Result", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
