import cv2
import os
import mediapipe as mp

output_dir = 'D:/ML/Data'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

#read image
img = cv2.imread('D:/ML/Data/face.jpeg')

H,W,_ = img.shape

#detect faces
mp_face_detection = mp.solutions.face_detection

#model_selection= 0  means it is best for faces within 2 meters from the camera
with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    out = face_detection.process(img_rgb)

    if out.detections is not None:
        for detection in out.detections:
            location_data = detection.location_data
            bbox = location_data.relative_bounding_box

            x1,y1,w,h = bbox.xmin, bbox.ymin, bbox.width, bbox.height

            x1 = int(x1*W)
            y1 = int(y1*H)
            w = int(w*W)
            h = int(h*H)

            #img = cv2.rectangle(img, (x1,y1),(x1+w,y1+h),(0,255,0),7)

            #blur faces
            img[y1:y1+h, x1:x1+w, :] = cv2.blur(img[y1:y1+h, x1:x1+w, :], (10,10))

    #cv2.imshow('img',img)
    #cv2.waitKey(0)

#save image
cv2.imwrite(os.path.join(output_dir, 'output.png'), img)