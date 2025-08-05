import cv2
import numpy as np
import matplotlib.pyplot as plt
import keyboard

# img = cv2.imread(r"C:\Users\novem\Downloads\me.jpg", flags=0)
# h, w = img.shape
# newH = h//4
# newW = w//4
# img = cv2.resize(img, (newH, newW))


# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     img_blur = cv2.GaussianBlur(gray,(3,3), sigmaX=0, sigmaY=0) 
#     edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200) 
 
#     # Display Canny Edge Detection Image
#     cv2.imshow('Canny Edge Detection', edges)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

def edge_detection(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(gray,(3,3), sigmaX=0, sigmaY=0) 
    edges = cv2.Canny(image=img_blur, threshold1=40, threshold2=100) 
    return edges
    

while True:
    if keyboard.is_pressed('q'):
        cam = cv2.VideoCapture(0)

        ret, frame = cam.read()

        if ret:
            cv2.imshow('Edge Detection', edge_detection(frame))     
            cv2.waitKey(0)                      
            cv2.destroyWindow("Captured")      
        else:
            print("Failed to capture image.")

        cam.release() 


