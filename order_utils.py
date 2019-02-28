from flask_wtf import Form
from wtforms import SelectField
from coffee_codes import *
from scipy.spatial import distance
import numpy as np
import sqlite3 
from statistics import mode
from statistics import StatisticsError

class ChooseForm(Form):
    blendselector = SelectField('Blend', choices=[(key,value) for key,value in blends.items()], default=1)
    typeselector = SelectField('Type', choices=[(key,value) for key,value in types.items()], default=1)
    sweetnessselector = SelectField('Sweetness', choices=[(key,value) for key,value in sweetness.items()], default=1)
    creamselector = SelectField('Type', choices=[(key,value) for key,value in cream.items()], default=1)
    milkselector = SelectField('Type', choices=[(key,value) for key,value in milk.items()], default=1)
    chocolateselector = SelectField('Type', choices=[(key,value) for key,value in chocolate.items()], default=1)
    toppingselector = SelectField('Type', choices=[(key,value) for key,value in topping.items()], default=1)
    blendselector1 = SelectField('Blend', choices=[(key,value) for key,value in blends.items()], default=1)
    typeselector1 = SelectField('Type', choices=[(key,value) for key,value in types.items()], default=1)
    sweetnessselector1 = SelectField('Sweetness', choices=[(key,value) for key,value in sweetness.items()], default=1)
    creamselector1 = SelectField('Cream', choices=[(key,value) for key,value in cream.items()], default=1)
    milkselector1 = SelectField('Milk', choices=[(key,value) for key,value in milk.items()], default=1)
    chocolateselector1 = SelectField('Chocolate', choices=[(key,value) for key,value in chocolate.items()], default=1)
    toppingselector1 = SelectField('Topping', choices=[(key,value) for key,value in topping.items()], default=1)
    

def get_bit_string(row):
    '''
    Converts the customer profile into a string of 9 bits first 3 bits personality, next 3 favorite 
    coffee accompaniment,last 3 favorite flavour - one bit in each category is set to true based on user enrollment info
    '''
    string = np.zeros(9)
    string[0+(row[2]-1)] = 1
    string[3+(row[3]-1)] = 1
    string[6+(row[4]-1)] = 1
    return string

def hamming_distance(str1, str2):
    '''
    Lesser the hamming distance more "nearest" the customers
    '''
    return distance.hamming(str1, str2)

def get_recomm(userid, conn, neighbors=5):
    cur = conn.cursor()
    cur.execute('SELECT * FROM Customer where id = ?',(userid,))
    row = cur.fetchone()
    src = get_bit_string(row)  
    cur.close()
    cur = conn.cursor()
    cur.execute('SELECT * FROM Customer')
    d={}
    for row in cur:
        if row[0] == userid: continue
        dest = get_bit_string(row)
        d[row[0]] = hamming_distance(src, dest)  
    cur.close()
    cur = conn.cursor()
    s = [x for x,_ in sorted(d.items(), key=lambda x: x[1], reverse=False)[:neighbors]] #get 5 "nearest neighbors"
    statement = "SELECT blend FROM History WHERE id IN ({0}) and rating IN (4,5)".format(', '.join(['?'] * len(s)))
    cur.execute(statement, s)
    try:
        print("here")
        return blends[mode([row[0] for row in cur])]
    except StatisticsError as e:
        try:#if there is no mode
            print("there")
            return blends[[row[0] for row in cur][0]]
        except IndexError:#if there are no neighbors who liked any of the coffees
            print("or not")
            return blends[1]
                
def save_order(userid, conn, order):
    #userid = 8
    print('userid ',userid)
    print('order  ',order)
    if not order:  # if order is not non-empty!!
        cur = conn.cursor()
        cur.execute('SELECT * FROM History WHERE id = ? order by datetime(datetime) DESC LIMIT 1', (userid, ))
        row  = cur.fetchone()
        order = {'blend':row[2],
                  'type':row[3],
                  'sweetness':row[4],
                  'cream':row[5],
                  'milk':row[6],
                  'chocolate':row[7],
                  'topping':row[8]}
        cur.close()
    cur = conn.cursor()
    cur.execute("insert into History (datetime, id, blend, type, sweetness, cream, milk, chocolate, topping) values \
                (datetime('now'),?,?,?,?,?,?,?,?)", 
                (userid,order['blend'],order['type'],order['sweetness'],order['cream'],order['milk'],order['chocolate'],
                order['topping']))
    cur.close()
    conn.commit()
    return 
