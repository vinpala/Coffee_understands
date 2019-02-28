from keras.models import Sequential
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform
from keras.engine.topology import Layer
from keras import backend as K
import cv2
import os
import numpy as np
import codecs
import sqlite3
import io
from numpy import genfromtxt
import pandas as pd
import tensorflow as tf
from fr_utils import *
from order_utils import *
from enroll_utils import *
from rating_utils import *
from inception_blocks_v2 import *
from coffee_codes import *
from loss import triplet_loss
from flask import Flask, render_template, request, session, flash, redirect, url_for
from imutils.video import VideoStream
import imutils
import time
import cv2
import os
import traceback
from collections import Counter
from format_html import generate_response_recognized, generate_response_not_recognized

app = Flask(__name__)

app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'

@app.route("/")
def detect():
    vs = VideoStream(src=0).start()
    time.sleep(1.0)
    found = False
    name_id = None
    total = 0
    names = []
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
        SSDmodel.setInput(blob)
        detections = SSDmodel.forward()
        print(detections)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):break
        # loop over the detections
        for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the prediction
            confidence = detections[0, 0, i, 2]
            #print("confidence:",confidence)
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
            #cv2.namedWindow('I see You!', cv2.WINDOW_NORMAL)
            cv2.imshow("I see you!", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"): break
    # if the `k` key was pressed, write the *original* frame to disk
    # so we can later process it and use it for face recognition
            if (total < 10):
                try:
                    face = frame[startY:startY+(endY-startY), startX:startX+(endX-startX)]
                    rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                    rgb = cv2.resize(rgb, (96, 96)) 
                    p = 'datasets/'+'visitor/'+"{}.png".format(str(total).zfill(5))
                    cv2.imwrite(p, rgb)
                    visitor_en = img_to_encoding(p, FRmodel)
                    name_id = who_is_it(visitor_en)
                    if name_id:
                        found=True
                        print("Found : ",name_id)
                        names.append(name_id) 
                    total += 1
                except Exception as e:
                    print("Exception !!",e)
                    print(traceback.format_exc())
            elif (not found): 
                cv2.destroyAllWindows()
                vs.stop()
                print("Do not recognize this face")
                session['userid'] = ''
                session['order'] ={}
                return render_template('found.html', 
                                           input=generate_response_not_recognized(), found=False)
         
            else:
                most_common,num_most_common = Counter(names).most_common(1)[0]
                if num_most_common > 5:
                    cv2.destroyAllWindows()
                    vs.stop()
                    session['userid'] = most_common
                    session['order'] ={}
                    if is_checkout(most_common):
                        return redirect(url_for('rating'))
                    else:    
                        return render_template('found.html', 
                                           input=generate_response_recognized(most_common, conn),found=True)
                else:
                    cv2.destroyAllWindows()
                    vs.stop()
                    session.pop('userid', None)
                    session.pop('order', None)
                    return render_template('found.html', 
                                           input=generate_response_not_recognized(), found=False)
                
@app.route('/decide', methods=['GET', 'POST'])
def decide():
    if request.method == 'POST':
        if "order" in request.form:
            session['order'] = {}
            return redirect(url_for('order'))
        elif "choose" in request.form:
            form = ChooseForm()
            recomm = get_recomm(session['userid'], conn)
            return render_template('choose.html', form=form, recommendation=recomm)
        else: 
            if "enroll" in request.form:
                form = EnrollmentForm()
                return render_template('enroll.html', form=form)
    
@app.route('/choose', methods=['GET', 'POST'])
def choose():
    if request.method == 'POST':
        #populate_order(session['userid'],session['order'])
        #redirect order
        if request.form['choice'] == 'choice1': #no recommender required 
            blend = int(request.form.get('blendselector'))
            type =  int(request.form.get('typeselector')) 
            sweetness = int(request.form.get('sweetnessselector')) 
            cream = int(request.form.get('creamselector')) 
            milk = int(request.form.get('milkselector'))
            chocolate = int(request.form.get('chocolateselector'))
            topping = int(request.form.get('toppingselector'))
            
        else:
            if request.form['choice'] == 'choice2':#give recommendation
                blend = int(request.form.get('blendselector1'))
                type =  int(request.form.get('typeselector1')) 
                sweetness = int(request.form.get('sweetnessselector1')) 
                cream = int(request.form.get('creamselector1')) 
                milk = int(request.form.get('milkselector1'))
                chocolate = int(request.form.get('chocolateselector1'))
                topping = int(request.form.get('toppingselector1'))
            else:
                flash('You have to choose an option', 'error')
                form = ChooseForm()
                recomm = get_recomm(session['userid'], conn)
                return render_template('choose.html', form=form, recommendation=recomm)
                 
        session['order'] = {'blend':blend,
                  'type':type,
                  'sweetness':sweetness,
                  'cream':cream,
                  'milk':milk,
                  'chocolate':chocolate,
                  'topping':topping}
        return redirect(url_for('order'))
        
@app.route('/rating', methods=['GET', 'POST'])
def rating():
    if request.method == 'POST':
        rating = request.form['rating'] 
        save_rating(session['userid'], rating, conn)
        flash('Thank you ! I will keep this in mind', 'success')
        time.sleep(1.0)
        return redirect(url_for('detect'))
    else:
        name, order = get_order_details(session['userid'])
        form = RatingForm()
        return render_template('rating.html', form=form, name=name, order=order)
                
@app.route('/order', methods=['GET', 'POST'])
def order():
    save_order(session['userid'], conn, session['order'])
    flash('Your order has been placed successfully!', 'success')
    time.sleep(1.0)
    return render_template('order.html')

@app.route('/moveon', methods=['GET', 'POST'])
def moveon():
    return redirect(url_for('detect'))

        
@app.route('/enroll', methods=['GET', 'POST'])
def enroll():
    if (request.method == 'POST'): 
        user={}
        user['name'] = request.form['name']
        user['personality'] = int(request.form.get('personality'))
        user['accompaniment'] = int(request.form.get('accompaniment'))
        user['flavors'] = int(request.form.get('flavors'))
        userid = save_user(user,conn)
        session['userid'] = userid
        flash('Give me a minute to bask in your beauty', 'success')
        enroll_face(userid, SSDmodel)
        images = [image for image in os.listdir('datasets/'+str(userid)+'/') if image.endswith(('.jpeg', '.png'))]
        cur = conn.cursor()
        for image in images:
            encoding = img_to_encoding('datasets/'+str(userid)+'/'+image, FRmodel)
            try:
                cur.execute("insert into Encoding (id, encoding) values (?,?)", (userid,encoding))
            except Exception as e:
                print("Exception !!",e)
                print(traceback.format_exc())         
        conn.commit()
        flash('Your face has been etched in my memory, forever, i think', 'success')
        form = ChooseForm()
        recomm = get_recomm(session['userid'], conn)
        return render_template('choose.html', form=form, recommendation=recomm)    
    
def adapt_array(arr):
    """
    http://stackoverflow.com/a/31312102/190597 (SoulNibbler)
    """
    # zlib uses similar disk size that Matlab v5 .mat files
    # bz2 compress 4 times zlib, but storing process is 20 times slower.
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    #return sqlite3.Binary(out.read().encode(compressor))  # zlib, bz2
    return sqlite3.Binary(codecs.encode(out.read(),compressor))

def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    #out = io.BytesIO(out.read().decode(compressor))
    out = io.BytesIO(codecs.decode(out.read(),compressor))
    return np.load(out)

def conn_database():
    global compressor 
    compressor = 'zlib'  # zlib, bz2
    # Converts np.array to TEXT when inserting
    sqlite3.register_adapter(np.ndarray, adapt_array)

    # Converts TEXT to np.array when selecting
    sqlite3.register_converter("array", convert_array)
    global conn
    conn = sqlite3.connect("coffee.sqlite", detect_types=sqlite3.PARSE_DECLTYPES)
    print("Database connected")


def load_models():
#The pretrained model for face recognition use Inception architecture takes 96x96 RBG images as input and outputs a 128-dimentional encoding.
#this uses channel-first convention so we need to tell Keras that.
    K.set_image_data_format('channels_first')
    global FRmodel
    FRmodel = faceRecoModel(input_shape=(3, 96, 96))
    FRmodel.compile(optimizer = 'adam', loss = triplet_loss, metrics = ['accuracy'])
    print("Loading Face Recognition model..may take some time..")
    load_weights_from_FaceNet(FRmodel)
    print("Successfully loaded Face recognition Model!")
    global SSDmodel
    SSDmodel = cv2.dnn.readNetFromCaffe('deploy.prototxt.txt', 'res10_300x300_ssd_iter_140000.caffemodel')
    print("Successfully loaded Face Detector!")  

def who_is_it(visitor_encoding):
     # Initialize "min_dist" to a large value, say 100
    name_id = None 
    min_dist = 100
    cur1 = conn.cursor()
    cur1.execute('SELECT * FROM Encoding')
    for row in cur1:
       
        # Compute L2 distance between the target "encoding" and the current "emb" from the database. 
        dist = np.linalg.norm(visitor_encoding - row[1])
        print(dist)
        
        # If this distance is less than the min_dist, then set min_dist to dist, and identity to name. 
        if dist < min_dist:
            min_dist = dist
            name_id = row[0]
          
    if min_dist > 0.7:
        print("Not in the database.")
        name_id = None
    cur1.close()

    return name_id

def is_checkout(name_id):
    cur = conn.cursor()
    cur.execute('Select * from History where id = ?and substr(datetime,1,10) = date("now") and rating is NULL', (name_id, ))
    row = cur.fetchone()
    cur.close()
    if row:
        return True
    else:
        return False

def get_order_details(name_id):
    cur = conn.cursor()
    cur.execute('Select * from History where id = ? and substr(datetime,1,10) = date("now") and rating is NULL LIMIT 1', (name_id, ))
    order ={}
    row = cur.fetchone()
    order['blend'] = blends[row[2]]
    order['type'] = types[row[3]]
    cur.close()
    cur1 = conn.cursor()
    cur1.execute('SELECT name FROM Customer WHERE id = ? ', (name_id, ))
    name = cur1.fetchone()[0]
    cur1.close()
    return name, order

if __name__ == "__main__":

    print("Loading models and Flask starting server..please wait until server has fully started")

    load_models()
    conn_database()

    # Run app

    app.run(host='0.0.0.0', port=50000)