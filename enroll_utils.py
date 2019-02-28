from flask_wtf import Form
from wtforms import TextField, RadioField
from coffee_codes import *
import sqlite3 
from imutils.video import VideoStream
import imutils
import time
import cv2
import os
import traceback
import numpy as np
from wtforms import validators, ValidationError

class EnrollmentForm(Form):
    name = TextField("Your Name",[validators.Required("Please enter your name.")])
    personality = RadioField('How would you describe yourself?', choices = [(key,value) for key,value in personality.items()])
    accompaniment = RadioField('What goes best with your coffee?', choices = [(key,value) for key,value in accompaniment.items()])
    flavors = RadioField('Which flavor do you like the most?', choices = [(key,value) for key,value in flavors.items()])

def save_user(user, conn):
    cur = conn.cursor()
    #id is the primary key it is autoincremented
    cur.execute("insert into Customer (name, personality, coffeewhen, tastes) values (?,?,?,?)", 
                (user['name'],user['personality'],user['accompaniment'],user['flavors']))
    userid = cur.lastrowid
    cur.close()
    conn.commit()
    return userid

def enroll_face(userid, detector):
    vs = VideoStream(src=0).start()
    time.sleep(1.0)
    found = False
    name_id = None
    total = 0
    names = []
    stop = False
    newpath = 'datasets/'+str(userid)
    if not os.path.exists(newpath): os.makedirs(newpath)
    # loop over the frames from the video stream

    while True:
        # grab the frame from the threaded video stream and resize it
        # to have a maximum width of 400 pixels
        frame = vs.read()
        frame = imutils.resize(frame, width=400)

        # grab the frame dimensions and convert it to a blob
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

        # pass the blob through the network and obtain the detections and predictions
        detector.setInput(blob)
        detections = detector.forward()
        print(detections)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):break
        # loop over the detections
        for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the prediction
            confidence = detections[0, 0, i, 2]
            print("confidence:",confidence)
        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
            if confidence < 0.5:
                continue

        # compute the (x, y)-coordinates of the bounding box for the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h]) 
            (startX, startY, endX, endY) = box.astype("int")
        # draw the bounding box of the face along with the associated probability
            text = "{:.2f}%".format(confidence * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10

            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or stop : break
    # if the `k` key was pressed, write the *original* frame to disk
    # so we can later process it and use it for face recognition
            if (total < 10):
                try:
                    face = frame[startY:startY+(endY-startY), startX:startX+(endX-startX)]
                    rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                    rgb = cv2.resize(rgb, (96, 96)) 
                    p = 'datasets/'+str(userid)+'/'+"{}.png".format(str(total).zfill(5))
                    cv2.imwrite(p, rgb)
                    total += 1
                except Exception as e:
                    print("Exception !!",e)
                    print(traceback.format_exc())

            else:
                cv2.destroyAllWindows()
                vs.stop() 
                return