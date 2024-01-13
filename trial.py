import cv2
import os
import numpy as np
import face_recognition
from attendance_utils import * 

# # Read the image
img_folder =[]
image_path = 'D:/PycharmProject/attendance_project_code/group_db/family_group_1.jpeg'  # Replace 'your_image_path.jpg' with the path to your image
# Resize the image to fit within the screen dimensions
# for img_path in os.listdir(image_path):
#     resized_image = cv2.imread(os.path.join(image_path,img_path))
#     resized_image = cv2.resize(resized_image, None, fx=0.1, fy=0.1)
#     location = face_recognition.face_locations(resized_image)
#     print(location)
#     try:
#         (y,m,n,x) =location[0]
#         img = cv2.rectangle(resized_image,(x,y),(m,n),(0,255,0),2)
#         cv2.imshow(img_path.split('.')[0],img)
#         img_folder.append(img)
#     except:
#         print(img_path)
#     # Display the resized image
    
# stacked_images = np.hstack(img_folder)
# cv2.imshow('Resized Image', stacked_images)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

CONF_THRESHOLD = 0.8
NMS_THRESHOLD = 0.4
IMG_WIDTH = 608
IMG_HEIGHT = 608

model_cfg ='D:/PycharmProject/attendance_project_code/models/yolov3-face.cfg'
model_weights = 'D:/PycharmProject/attendance_project_code/models/yolov3-wider_16000.weights'
net = cv2.dnn.readNetFromDarknet(model_cfg,model_weights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

cap = cv2.VideoCapture(image_path)
while True:
    has_frame, frame = cap.read()
    if not has_frame:
        print('end')
        break
    print('start')
    blob = cv2.dnn.blobFromImage(frame, 1 / 255, (IMG_WIDTH, IMG_HEIGHT),[0, 0, 0], 1, crop=False)
    
    # Sets the input to the network
    net.setInput(blob)

    # Runs the forward pass to get output of the output layers
    outs = net.forward(get_outputs_names(net)) #in utils

    # Remove the bounding boxes with low confidence
    faces = post_process(frame, outs, CONF_THRESHOLD, NMS_THRESHOLD)

    print('[i] ==> # detected faces: {}'.format(len(faces)))
    print('#' * 60)

    cv2.imwrite('D:/PycharmProject/attendance_project_code/cropped_images/grp_id.jpg',frame)
    cv2.imshow('video',frame)
    key = cv2.waitKey(1)
    if key == 27 or key == ord('q'):
        print('[i] ==> Interrupted by user!')
        break
    
cv2.destroyAllWindows()