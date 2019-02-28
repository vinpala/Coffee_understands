import sqlite3
from coffee_codes import *
form = '<form method="post" action="/decide">'
# TODO - move more html layout stuff to templates and send the strings alone to Jinja  .. make this less clumsy!!

def generate_response_recognized(customer_id, conn):
    
    cur2 = conn.cursor()
    cur2.execute('SELECT name FROM Customer WHERE id = ? LIMIT 1', (customer_id, ))
    name  = cur2.fetchone()[0]
    cur = conn.cursor()
    cur.execute('SELECT * FROM History WHERE id = ? order by datetime(datetime) DESC LIMIT 1', (customer_id, ))
    row  = cur.fetchone()
    history = 'Welcome Back '+name+' !!' + '<p> Last time you were here you had a '+ blends[row[2]]+ ' '+types[row[3]]+ ' with ' + \
    sweetness[row[4]]+' with '+cream[row[5]]+' with '+milk[row[6]]+' with '+chocolate[row[7]]+' with '+topping[row[8]]+ '</p>'

    if row[9] in [4,5]: #satisfied customer
        rating_text = '<p> You seemed to like it, wanna go for that again ?</p>' +'<div class="row"> <div class="col-md-4">'
        rating_buttons = '<input type="submit" class="btn btn-primary btn-lg" name="order" value="YES!"/></input>'+\
        '</div>'+'<div class="col-md-4">'+'<input type="submit" class="btn btn-primary btn-lg" name="choose"' +\
        'value="No..something different this time"> </input>'+'</div></form> </div>'
    else:
        if row[9] in [1,2,3]: #pissed off customer
            rating_text = '<p> Sorry..it dint hit the spot for you, shall we try again ?</p>' +\
            '<div class="row"> <div class="col-md-4">'
        if not row[9]: #dint leave a rating
            rating_text = '<p> We are not sure what you thought about it,but now shall we get you your cuppa joe ?</p>' +\
            '<div class="row"> <div class="col-md-4">'
            
        rating_buttons = '<input type="submit" class="btn btn-primary btn-lg" name="choose" value="Yes!"></input>'+\
        '</div>'+'<div class="col-md-4">'+\
        '<input type="button" class="btn btn-primary btn-lg" name="submit_button" '+\
        'value="Oh no..I cant believe I came here again..this is the wrath of karma"></input> </div></form></div>'
    return history+rating_text+form+rating_buttons

def generate_response_not_recognized():
    html ='<p>Welcome to our cafe! Lets find you your coffee..</p> <div class="row"> <div class="col-md-4">'+form+\
     '<input type="submit" class="btn btn-primary btn-lg" name="enroll" value="Yes!"></input>'+'</div>'+\
        '<div class="col-md-4">'+\
    '<input type="button" class="btn btn-primary btn-lg" name="submit_button" '+\
    'value="Oh no..I thought this was my shrink\'s office.."></input> </div></form></div>'
    return html
                
 