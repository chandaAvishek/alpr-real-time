import cv2, torch, easyocr
print("OpenCV:", cv2.__version__)
print("Torch:", torch.__version__)
reader = easyocr.Reader(['en'])
print("EasyOCR Ready:", reader is not None)
