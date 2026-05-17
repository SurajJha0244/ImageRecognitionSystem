from flask import Flask, request, render_template
import face_recognition
import pickle
import os
import numpy as np
import uuid

app = Flask(__name__)

TRAIN_DIR = "train"
MODEL_PATH = "model.pkl"

os.makedirs(TRAIN_DIR, exist_ok=True)

# LOAD MODEL
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
else:
    model = {
        "encodings": [],
        "names": []
    }


# RETRAIN MODEL FUNCTION
def retrain_model():

    known_encodings = []
    known_names = []

    for file_name in os.listdir(TRAIN_DIR):

        if file_name.endswith((".jpg", ".png", ".jpeg")):

            image_path = os.path.join(TRAIN_DIR, file_name)

            image = face_recognition.load_image_file(image_path)

            encodings = face_recognition.face_encodings(image)

            if len(encodings) > 0:

                encoding = encodings[0]

                name = os.path.splitext(file_name)[0]

                known_encodings.append(encoding)
                known_names.append(name)

    new_model = {
        "encodings": known_encodings,
        "names": known_names
    }

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(new_model, f)

    return new_model


# HOME PAGE
@app.route("/")
def home():
    return '''
<!DOCTYPE html>
<html>
<head>
    <title>Face Recognition System</title>

    <style>
        body {
            margin: 0;
            font-family: Arial;
            background: linear-gradient(135deg, #74ebd5, #ACB6E5);
            height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
        }

        .container {
            text-align: center;
            background: white;
            padding: 50px 40px;
            border-radius: 15px;
            box-shadow: 0px 10px 30px rgba(0,0,0,0.2);
            width: 350px;
        }

        h1 {
            margin-bottom: 30px;
            color: #2c3e50;
        }

        a {
            display: block;
            margin: 15px 0;
            padding: 12px;
            text-decoration: none;
            background: #3498db;
            color: white;
            border-radius: 8px;
            font-size: 16px;
            transition: 0.3s;
        }

        a:hover {
            background: #217dbb;
        }

    </style>
</head>

<body>

<div class="container">

    <h1>Face Recognition System</h1>

    <a href="/train_page">Train Image</a>
    <a href="/recognize_page">Recognize Image</a>

</div>

</body>
</html>
'''


# TRAIN PAGE
@app.route("/train_page")
def train_page():
    return render_template("train.html")


# RECOGNIZE PAGE
@app.route("/recognize_page")
def recognize_page():
    return render_template("recognize.html")


# TRAIN MODEL
@app.route("/train", methods=["POST"])
def train():

    global model

    name = request.form["person_name"]
    file = request.files["image"]

    if file.filename == "":
        return render_template(
            "train.html",
            error="No file selected"
        )

# Safe unique filename
    extension = os.path.splitext(file.filename)[1]

    unique_filename = f"{name}_{len(model['names'])}{extension}"
    image_path = os.path.join(TRAIN_DIR, unique_filename)

    file.save(image_path)

# Load image
    image = face_recognition.load_image_file(image_path)
    encodings = face_recognition.face_encodings(image)

    if len(encodings) == 0:
        os.remove(image_path)
        return render_template(
            "train.html",
            error="No face found in image"
        )

    encoding = encodings[0]

# DUPLICATE FACE CHECK
    if len(model["encodings"]) > 0:

        distances = face_recognition.face_distance(
            model["encodings"],
            encoding
        )

        best_index = distances.argmin()

# same face detected
        if distances[best_index] < 0.5:

            existing_name = model["names"][best_index]

            return render_template(
                "train.html",
                duplicate=True,
                old_name=existing_name,
                new_name=name,
                encoding=encoding.tolist(),
                image_path=image_path
            )

# ADD NEW FACE
    model["encodings"].append(encoding)
    model["names"].append(name)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    return render_template(
        "train.html",
        success="Trained Successfully",
        name=name
    )

# Update new face    
@app.route("/update_face", methods=["POST"])
def update_face():

    global model

    old_name = request.form["old_name"]
    new_name = request.form["new_name"]

    updated = False

    for i in range(len(model["names"])):
        if model["names"][i] == old_name:
            model["names"][i] = new_name
            updated = True

    if not updated:
        return render_template(
            "train.html",
            error="No matching face found to update"
        )

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    return render_template(
        "train.html",
        success="Name Updated Successfully",
        name=new_name
    )

# RECOGNIZE FACE
@app.route("/recognize", methods=["POST"])
def recognize():

    file = request.files["image"]

    image = face_recognition.load_image_file(file)

    encodings = face_recognition.face_encodings(image)

    if len(encodings) == 0:
        return render_template(
            "recognize.html",
            error="No face found"
        )

    test_encoding = encodings[0]

    # Better matching
    face_distances = face_recognition.face_distance(
        model["encodings"],
        test_encoding
    )

    if len(face_distances) == 0:
        return render_template(
            "recognize.html",
            error="No trained faces found"
        )

    best_index = np.argmin(face_distances)

    # Threshold
    if face_distances[best_index] < 0.5:

        name = model["names"][best_index]

        return render_template(
            "recognize.html",
            success="Match Found",
            name=name
        )

    return render_template(
        "recognize.html",
        error="No Match Found"
    )


# DELETE PERSON
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


# RUN APP
if __name__ == "__main__":
    app.run(debug=True)