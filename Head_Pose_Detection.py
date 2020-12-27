import cv2
import numpy as np
import math
from FaceDetection import face_detector_model, find_faces, draw_boxes
from MarkDetector import face_landmark_model, detect_marks

#Function to return the 3D points present as 2D for making annotation box
def get_2d_points(img, rotation_vector, translation_vector, camera_matrix, val):

    point_3d = []
    dist_coeffs = np.zeros((4,1))
    rear_size = val[0]
    rear_depth = val[1]
    point_3d.append((-rear_size, -rear_size, rear_depth))
    point_3d.append((-rear_size, rear_size, rear_depth))
    point_3d.append((rear_size, rear_size, rear_depth))
    point_3d.append((rear_size, -rear_size, rear_depth))
    point_3d.append((-rear_size, -rear_size, rear_depth))

    front_size = val[2]
    front_depth = val[3]
    point_3d.append((-front_size, -front_size, front_depth))
    point_3d.append((-front_size, front_size, front_depth))
    point_3d.append((front_size, front_size, front_depth))
    point_3d.append((front_size, -front_size, front_depth))
    point_3d.append((-front_size, -front_size, front_depth))
    point_3d = np.array(point_3d, dtype=np.float).reshape(-1, 3)

    #Map to 2d img points
    (point_2d, _) = cv2.projectPoints(point_3d,
                                      rotation_vector,
                                      translation_vector,
                                      camera_matrix,
                                      dist_coeffs)
    point_2d = np.int32(point_2d.reshape(-1, 2))
    return point_2d

#Function to get points to identify side head-pose. Returns a tuple of coordinates of line to estimate head-pose.
def head_pose_points(img, rotation_vector, translation_vector, camera_matrix):

    rear_size = 1
    rear_depth = 0
    front_size = img.shape[1]
    front_depth = front_size*2
    val = [rear_size, rear_depth, front_size, front_depth]
    point_2d = get_2d_points(img, rotation_vector, translation_vector, camera_matrix, val)
    y = (point_2d[5] + point_2d[8])//2
    x = point_2d[2]

    return (x, y)
    
face_model = face_detector_model()
landmark_model = face_landmark_model()
cap = cv2.VideoCapture(0)
ret, img = cap.read()
size = img.shape
font = cv2.FONT_HERSHEY_PLAIN
# 3D model points.
model_points = np.array([
                            (0.0, 0.0, 0.0),             # Nose tip
                            (0.0, -330.0, -65.0),        # Chin
                            (-225.0, 170.0, -135.0),     # Left eye left corner
                            (225.0, 170.0, -135.0),      # Right eye right corne
                            (-150.0, -150.0, -125.0),    # Left Mouth corner
                            (150.0, -150.0, -125.0)      # Right mouth corner
                        ])

# Camera internals
focal_length = size[1]
center = (size[1]/2, size[0]/2)
camera_matrix = np.array(
                         [[focal_length, 0, center[0]],
                         [0, focal_length, center[1]],
                         [0, 0, 1]], dtype = "double"
                         )
while True:
    ret, img = cap.read()
    if ret == True:
        faces = find_faces(img, face_model)
        for face in faces:
            marks = detect_marks(img, landmark_model, face)
            image_points = np.array([
                                    marks[30],     #Nose tip
                                    marks[8],      #Chin
                                    marks[36],     #Left eye left corner
                                    marks[45],     #Right eye right corner
                                    marks[48],     #Left Mouth corner
                                    marks[54]      #Right mouth corner
                                ], dtype="double")
            dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
            (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_UPNP)
            
            
            # Project a 3D point (0, 0, 1000.0) onto the image plane. (draw line sticking out of the plane)
            
            (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
            
            for p in image_points:
                cv2.circle(img, (int(p[0]), int(p[1])), 3, (0,255,0), -1)
            
            
            p1 = ( int(image_points[0][0]), int(image_points[0][1]))
            p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
            x1, x2 = head_pose_points(img, rotation_vector, translation_vector, camera_matrix)

            cv2.line(img, p1, p2, (255, 0, 0), 1)
            cv2.line(img, tuple(x1), tuple(x2), (0, 0, 255), 1)

            try:
                m = (p2[1] - p1[1])/(p2[0] - p1[0])
                ang1 = int(math.degrees(math.atan(m)))
            except:
                ang1 = 90
                
            try:
                m = (x2[1] - x1[1])/(x2[0] - x1[0])
                ang2 = int(math.degrees(math.atan(-1/m)))
            except:
                ang2 = 90
                
            if ang1 >= 48:
                print('Head down')
                cv2.putText(img, 'Head down', (30, 30), font, 2, (0, 0, 0), 2)
            elif ang1 <= -48:
                print('Head up')
                cv2.putText(img, 'Head up', (30, 30), font, 2, (0, 0, 0), 2)
             
            if ang2 >= 48:
                print('Head right')
                cv2.putText(img, 'Head right', (90, 30), font, 2, (0, 0, 0), 2)
            elif ang2 <= -48:
                print('Head left')
                cv2.putText(img, 'Head left', (90, 30), font, 2, (0, 0, 0), 2)
            
            cv2.putText(img, str(ang1), tuple(p1), font, 2, (255, 0, 0), 2)
            cv2.putText(img, str(ang2), tuple(x1), font, 2, (0, 0, 255), 2)
            draw_boxes(img,face)
        cv2.imshow('Head Pose Detector', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
cv2.destroyAllWindows()
cap.release()
