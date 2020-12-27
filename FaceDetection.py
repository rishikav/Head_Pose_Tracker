import cv2
import numpy as np

#Function to return face detection caffemodel of opencv's dnn module
def face_detector_model():

    modelFile = "models/res10_300x300_ssd_iter_140000.caffemodel"
    configFile = "models/deploy.prototxt"
    model = cv2.dnn.readNetFromCaffe(configFile, modelFile)
    return model

#Function to return list of coordinates of faces detected.
def find_faces(img, model):

    h, w = img.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    model.setInput(blob)
    res = model.forward()
    faces = []
    for i in range(res.shape[2]):
        confidence = res[0, 0, i, 2]
        if confidence > 0.5:
            box = res[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x, y, x1, y1) = box.astype("int")
            faces.append([x, y, x1, y1])
    return faces

#Function to draw bounding boxes around the faces in the frame.
def draw_boxes(img, faces):

    cv2.rectangle(img, (faces[0], faces[1]), (faces[2], faces[3]), (0, 255, 0), 3)



