from flask_wtf import Form
from wtforms import RadioField
from coffee_codes import *
import sqlite3 

class RatingForm(Form):
    rating = RadioField('How did you like your Coffee?', choices = [(key,value) for key,value in rating.items()])

def save_rating(user_id, rating, conn):
    cur = conn.cursor()
    cur.execute('Select datetime from History where id = ? and substr(datetime,1,10) = date("now") and rating is NULL LIMIT 1', (user_id, ))
    key = cur.fetchone()[0]
    cur.close()
    cur = conn.cursor()
    cur.execute("update History set rating=? where datetime= ?",(rating,key))
    conn.commit()
    return
