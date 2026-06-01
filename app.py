from flask import Flask, request, render_template, redirect, session
import os
from flask_sqlalchemy import SQLAlchemy
import bcrypt
import re

from train_model import (
    load_model,
    train_face,
    retrain_model,
    update_face_name
)

from recognize import recognize_face

app = Flask(__name__)

BASE_DIR=os.path.abspath(os.path.dirname(__file__))
db_path=os.path.join(BASE_DIR,"database","database.db")

# Database Connection
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SECRET_KEY'] = 'thisisasecretkey'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
app.secret_key = 'thisisasecretkey'


# Creating Database 

class User(db.Model):
    id=db.Column(db.Integer,primary_key=True)
    username=db.Column(db.String(20),unique=True,nullable=False)
    password=db.Column(db.String(80),nullable=False)

    def __init__(self,username,password):
        self.username = username
        self.password = bcrypt.hashpw(password.encode('utf-8'),bcrypt.gensalt()).decode('utf-8')

    def check_password(self,password):
        return bcrypt.checkpw(password.encode('utf-8'),self.password.encode('utf-8')) 

with app.app_context():
     db.create_all()
print("database created successfully ")
print(db_path)     

model = load_model()

TRAIN_DIR = "train"


# HOME PAGE
@app.route('/dashboard')
def dashboard():

    if not session.get('username'):   
        return redirect('/login_page')

    return render_template('dashboard.html')



# Login PAGE
@app.route("/login_page", methods=['GET','POST'])
def login_page():

    if request.method == "POST":

        username = request.form['username']
        password = request.form['password']

        user = User.query.filter_by(username=username).first()

        if user:
            print("USER FOUND")

            if user.check_password(password):
                print("LOGIN SUCCESS")
                session.clear()
                session['username'] = user.username
                return redirect('/dashboard')

        print("LOGIN FAILED")
        return render_template('login.html', error="Invalid user")

    return render_template('login.html')


# Register Page
@app.route("/",methods=['GET','POST'])

def register_page():
    if request.method == "POST":
     # Handle request 
     username = request.form['username']
    
     password = request.form['password']

# Username validation
     if not re.match("^[a-zA-Z0-9]+$", username):
         return render_template(
                'register.html',
                error="Username must be alphanumeric without spaces"
            )
# Check existing username
     existing_user = User.query.filter_by(username=username).first()

     if existing_user:
         return render_template(
                'register.html',
                error="Username already exists"
            )

     new_user = User(username=username,password=password)
     db.session.add(new_user)
     db.session.commit()
     return redirect('/login_page')


    return render_template('register.html')

# Logout Page
@app.route('/logout_page')
def logout_page():
    session.pop('username', None)

    return redirect('/login_page')

# Delete user By   username 

@app.route('/delete_user/<username>')
def delete_user(username):
     if not session.get('username'):
         return redirect('/login_page')
      

     user = User.query.filter_by(username=username).first()

     if user:
         db.session.delete(user)
         db.session.commit()

         return "User deleted successfully"

     return "User not found" 

# Forgot password

@app.route('/forget_password',methods=["GET","POST"])
def forget_password():
     if request.method == "POST":
         username = request.form["username"]
     
         user=User.query.filter_by(username=username).first()

         if user:
            return redirect(f"/reset_password/{username}")
         else:
            return "User not found"

     return render_template("forget_password.html")
     

# Reset Password
@app.route('/reset_password/<username>',methods=['GET','POST'])
def reset_password(username):

    user = User.query.filter_by(username=username).first()

    if request.method == "POST":
        new_password = request.form["password"]

        user.password = bcrypt.hashpw(
            new_password.encode('utf-8'),
            bcrypt.gensalt()
        ).decode('utf-8')

        db.session.commit()

        return redirect("/login_page")

    return render_template("reset_password.html")  


       




# TRAIN PAGE
@app.route("/train_page")
def train_page():
    return render_template("train.html")


# RECOGNIZE PAGE
@app.route("/recognize_page")
def recognize_page():
    return render_template("recognize.html")


# TRAIN
@app.route("/train", methods=["POST"])
def train():

    global model

    name = request.form["person_name"]

    file = request.files["image"]

    result = train_face(name, file, model)

    if result["status"] == "error":

        return render_template(
            "train.html",
            error=result["message"]
        )

    elif result["status"] == "duplicate":

        return render_template(
            "train.html",
            duplicate=True,
            old_name=result["old_name"],
            new_name=result["new_name"]
        )

    return render_template(
        "train.html",
        success="Trained Successfully",
        name=result["name"]
    )


# UPDATE FACE
@app.route("/update_face", methods=["POST"])
def update_face():

    global model

    old_name = request.form["old_name"]

    new_name = request.form["new_name"]

    updated = update_face_name(
        old_name,
        new_name,
        model
    )

    if updated:

        return render_template(
            "train.html",
            success="Name Updated Successfully",
            name=new_name
        )

    return render_template(
        "train.html",
        error="No matching face found"
    )


# RECOGNIZE
@app.route("/recognize", methods=["POST"])
def recognize():

    file = request.files["image"]

    result = recognize_face(file, model)

    if result["status"] == "success":

        return render_template(
            "recognize.html",
            success="Match Found",
            name=result["name"]
        )

    return render_template(
        "recognize.html",
        error=result["message"]
    )


# DELETE
@app.route("/delete/<name>")
def delete(name):

    global model

    extensions = [".jpg", ".png", ".jpeg"]

    deleted = False

    for ext in extensions:

        image_path = os.path.join(TRAIN_DIR, name + ext)

        if os.path.exists(image_path):

            os.remove(image_path)

            deleted = True
            break

    if deleted:

        model = retrain_model()

        return f"{name} deleted successfully"

    return "Image not found"


# RUN
if __name__ == "__main__":

    app.run(debug=True)