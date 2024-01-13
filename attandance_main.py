import cv2
import numpy as np
import face_recognition
import os
import shutil
import pandas as pd
from attendance_utils import *

def img_folder(path_db):
    images = []
    classNames = []
    myList = os.listdir(path_db)
    # print(myList)
    for cl in myList:
        curImg = cv2.imread(f'{path_db}/{cl}')
        curImg = cv2.resize(curImg,(400,400)) #to resize image to standard size 400x400
        images.append(curImg)
        classNames.append(os.path.splitext(cl)[0])
    
    return images,classNames

def findEncodings(images):
    encodeList = []
    count =0
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        count+=1
        try:
            encode = face_recognition.face_encodings(img)[0]
            print(count)
            encodeList.append(encode)
        except:
            pass
    return encodeList
 

def update_csv(csv_file, attendance_data):
    if not attendance_data:
        return
    df = pd.read_csv(csv_file)
    for name in attendance_data:
        df.loc[df['NAME'] == name, 'ATTENDANCE'] = "PRESENT"
    df.to_csv(csv_file, index=False)

def cropping_images_form_folder(path_db,path_crop):
    output_folder = path_crop
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for img_path in os.listdir(path_db):
        resized_image = cv2.imread(os.path.join(path_db,img_path))
        resized_image = cv2.resize(resized_image,(400,400))
        location = face_recognition.face_locations(resized_image)
        print(location)
        try:
            (top,right,bottom,left) =location[0]
            # resized_image = cv2.rectangle(resized_image,(left,top),(right,bottom),(0,255,0),2) 
            cv2.imshow(img_path.split('.')[0],cropped_image)
            cropped_image = resized_image[top:bottom, left:right]
            output_path = os.path.join(output_folder, f"cropped_image_{img_path}.jpg")
            cv2.imwrite(output_path,cropped_image )
        except:
            print(img_path)
        
def crop_photo_attendance():
    images_crop,classNames_crop=img_folder(path_crop)
    encodeListCrop = findEncodings(images_crop)
    print('Encoding Complete')
    present_student=[]
    for encodeFace in encodeListCrop:
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
        print(matches)
        faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
        #print(faceDis)
        matchIndex = np.argmin(faceDis)
        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(name)
            present_student.append(name)

    update_csv(path_csv,present_student)

def identify_face(path_main):
    cap = cv2.VideoCapture(path_main)
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
    

def video_attendance(): 
    cap = cv2.VideoCapture(0)
    
    while True:
        success, img = cap.read()
        imgS = cv2.resize(img,(0,0),None,0.25,0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
        
        facesCurFrame = face_recognition.face_locations(imgS)
        encodesCurFrame = face_recognition.face_encodings(imgS,facesCurFrame)
        present_student=[]
        for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
            #print(faceDis)
            matchIndex = np.argmin(faceDis)
            if matches[matchIndex]:
                name = classNames[matchIndex].upper()
                print(name)
                y1,x2,y2,x1 = faceLoc
                y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
                cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
                cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
                present_student.append(name)

        update_csv(path_csv,present_student)
        cv2.imshow('Webcam',img)
        if cv2.waitKey(1) == 27 or cv2.waitKey(1) == ord('q'):
            break


if __name__ == "__main__":
    path_db = 'D:/PycharmProject/attendance_project_code/family_db' #database of known image 
    path_crop = 'D:/PycharmProject/attendance_project_code/cropped_images' #cropped image from main
    path_csv = 'D:/PycharmProject/attendance_project_code/attendence.CSV' #attendance csv
    path_main = 'D:/PycharmProject/attendance_project_code/group_db/family_group_6.jpg' #main photo through which it should conduct the code

    model_cfg ='D:/PycharmProject/attendance_project_code/models/yolov3-face.cfg'
    model_weights = 'D:/PycharmProject/attendance_project_code/models/yolov3-wider_16000.weights'
    net = cv2.dnn.readNetFromDarknet(model_cfg,model_weights)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    
    CONF_THRESHOLD = 0.8
    NMS_THRESHOLD = 0.4
    IMG_WIDTH = 416
    IMG_HEIGHT = 416
    if os.path.exists(path_crop):
        shutil.rmtree(path_crop)

    identify_face(path_main)
    images,classNames=img_folder(path_db)
    encodeListKnown = findEncodings(images)
    print('Encoding Complete')
    crop_photo_attendance()
    # video_attendance()