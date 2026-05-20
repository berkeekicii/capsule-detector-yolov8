from ultralytics import YOLO
import cv2
import numpy as np

def letterbox_no_zoom(im, new_shape=(640, 640), color=(0, 0, 0)):
    h, w = im.shape[:2]

    # sadece küçült (büyütme yok!)
    r = min(new_shape[0] / h, new_shape[1] / w)
    r = min(r, 1.0)  # 👈 KRİTİK: büyütmeyi engeller

    new_w, new_h = int(w * r), int(h * r)
    im_resized = cv2.resize(im, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # padding
    pad_w = new_shape[1] - new_w
    pad_h = new_shape[0] - new_h

    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    im_padded = cv2.copyMakeBorder(
        im_resized, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=color
    )

    return im_padded


model = YOLO("D:/AI_Project/dataset/runs/detect/train-3/weights/best.pt")

img0 = cv2.imread("D:/AI_Project/test/foto1.jpg")

img = letterbox_no_zoom(img0, (640, 640))

results = model(img)

out = results[0].plot()

cv2.imshow("Result", out)
cv2.waitKey(0)
cv2.destroyAllWindows()