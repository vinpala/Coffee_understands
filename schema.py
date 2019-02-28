import sqlite3
con = sqlite3.connect("coffee.sqlite", detect_types=sqlite3.PARSE_DECLTYPES)
cur = con.cursor()
cur.execute('''CREATE TABLE IF NOT EXISTS Encoding
    (id INTEGER NOT NULL, encoding array)''')

cur.execute('''CREATE TABLE IF NOT EXISTS Customer
    (id INTEGER PRIMARY KEY, name TEXT, personality INTEGER, coffeewhen INTEGER, tastes INTEGER)''')

cur.execute('''CREATE TABLE IF NOT EXISTS History
    (datetime TEXT PRIMARY KEY, id INTEGER NOT NULL, blend INTEGER, type INTEGER, sweetness INTEGER, cream INTEGER, milk INTEGER, chocolate INTEGER, topping INTEGER, rating INTEGER)''')