import cv2
import numpy as np
import pytesseract
from PIL import Image
from django.conf import settings
import os 

STATIC_DIR = settings.STATIC_DIR

frontal_face = cv2.CascadeClassifier(os.path.join(STATIC_DIR,'./cascade_classifier/haarcascade_frontalface_default.xml'))
eye_class = cv2.CascadeClassifier(os.path.join(STATIC_DIR,'./cascade_classifier/haarcascade_eye.xml'))
eyel_class = cv2.CascadeClassifier(os.path.join(STATIC_DIR,'./cascade_classifier/haarcascade_lefteye_2splits.xml'))
eyer_class = cv2.CascadeClassifier(os.path.join(STATIC_DIR,'./cascade_classifier/haarcascade_righteye_2splits.xml'))
smile_dect = cv2.CascadeClassifier(os.path.join(STATIC_DIR,'./cascade_classifier/haarcascade_smile.xml'))
closedeyesspecs=cv2.CascadeClassifier(os.path.join(STATIC_DIR,'./cascade_classifier/haarcascade_eye_tree_eyeglasses.xml'))
lower_body_detect=cv2.CascadeClassifier(os.path.join(STATIC_DIR,'./cascade_classifier/haarcascade_lowerbody.xml'))

# "Closed_Eyes_Detected" : False,
assesment_dict = {"No_Faces_Detected" : False, "Multiple_Faces_Detected" : False, "Lower_Body_Detected" : False ,"Invalid_Face_Detected" : False,
                    "No_Eyes_Detected" : False, "No_Mouth_Detected" : False, "Text_Detected" : False, "Final_criteria" : True}

def facedetector(path):
    count = []
    lim= 1
    ## This code will handle the first three and the last point of the problem statement.
    ## Make changes in the print statements as per the I/O of the file. Use the cv2.circles and cv2.rectangles while
    ## cross verifying else just comment them.
    img = cv2.imread(path)
    cv2.imwrite(os.path.join(settings.MEDIA_ROOT,'ml_out/process.jpg'),img)
    faces, num_detection_face =frontal_face.detectMultiScale2(img,minNeighbors=20)
    number_of_faces = len(faces)
    if number_of_faces == 0:
        ######## Modification Left ###########
        ######## We have to exit our programme from here #########
        assesment_dict["No_Faces_Detected"] = True
        assesment_dict["Final_criteria"] = False
        return assesment_dict
    elif number_of_faces > 1:
        ######## Modification Left ###########
        ######## We have to exit our programme from here #########
        assesment_dict["Multiple_Faces_Detected"] = True
        assesment_dict["Final_criteria"] = False
        return assesment_dict
    # coord1,closedeyedetect=closedeyesspecs.detectMultiScale2(img)
    # if len(closedeyedetect) != 0:
    #     assesment_dict["Closed_Eyes_Detected"] = True
    coord2,lowerbodydetect=lower_body_detect.detectMultiScale2(img, minNeighbors = 10)
    if len(lowerbodydetect) != 0:
        assesment_dict["Lower_Body_Detected"] = True
    for x,y,w,h in faces:
        count.append(lim)
        face_roi = img[y:y+h,x:x+h].copy() # croping the image
        # cv2.imwrite(os.path.join(settings.MEDIA_ROOT,'ml_out/roi_{}.jpg'.format(lim)),face_roi)
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0))
            
    #apply to cascasde classifier (eyes)
        eyesl,num_detection_eyesl = eyel_class.detectMultiScale2(face_roi,minNeighbors = 5)
        eyesr,num_detection_eyesr = eyer_class.detectMultiScale2(face_roi,minNeighbors = 10)
        if len(eyesr) == 0 and len(eyesl) == 0:
            assesment_dict["No_Eyes_Detected"] = True
        for ex, ey, ew, eh in eyesl:
            cx = x+ex+ew//2
            cy = y+ey+eh//2
            if cx<x or cx>(x+w) or cy<y or cy>(y+w):
                assesment_dict["Invalid_Face_Detected"] = True
                #This is the condition for checking if eyes are outside the face XD
            r = eh //2
            #cv2.circle(img,(cx,cy),r,(255,0,255),2)
        for ex, ey, ew, eh in eyesr:
            cx = x+ex+ew//2
            cy = y+ey+eh//2
            if cx<x or cx>(x+w) or cy<y or cy>(y+w):
                assesment_dict["Invalid_Face_Detected"] = True
                #This is the condition for checking if eyes are outside the face XD
            #r = eh //2
            #cv2.circle(img,(cx,cy),r,(255,0,255),2)
        #apply to cascasde classifier (mouth)
        smiles , num_detection_smile = smile_dect.detectMultiScale2(face_roi,minNeighbors = 20)
        if len(smiles) == 0:
            assesment_dict["No_Mouth_Detected"] = True
#         for sx,sy, sw,sh in smiles:
#             #cv2.rectangle(img,(x+sx,y+sy),(x+sx+sw,y+sy+sh),(255,0,0),2)
        pytesseract.pytesseract.tesseract_cmd = '/usr/local/Cellar/tesseract/4.1.1/bin/tesseract'
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18))
        dilation = cv2.dilate(thresh1, rect_kernel, iterations = 1)
        contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        im2 = img.copy()
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            rect = cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cropped = im2[y:y + h, x:x + w]
            text = pytesseract.image_to_string(cropped)
            if bool(text.strip()) == True:
                assesment_dict["Text_Detected"] = True

        ## Final Detection : 
        for i in assesment_dict.keys():
            if assesment_dict[i] == True and i != "Final_criteria":
                assesment_dict["Final_criteria"] = False
        lim += 1
    assesment_dict["count"] = count
    return assesment_dict



