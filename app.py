from flask import Flask, request, jsonify, render_template
import face_recognition
import pickle
import os

app = Flask(__name__)

TRAIN_DIR = "train"
MODEL_PATH = "model.pkl"

os.makedirs(TRAIN_DIR, exist_ok=True)

# Load or init model
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
else:
    model = {"encodings": [], "names": []}



# HOME (optional)

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


# TRAIN API
@app.route("/train", methods=["POST"])
def train():

    file = request.files["image"]

    if file.filename == "":
        return render_template(
            "train.html",
            error="No file selected"
        )

    image_path = os.path.join(TRAIN_DIR, file.filename)
    file.save(image_path)

    image = face_recognition.load_image_file(image_path)

    encodings = face_recognition.face_encodings(image)

    if len(encodings) == 0:
        return render_template(
            "train.html",
            error="No face found in image"
        )

    encoding = encodings[0]

# Get name from input field
    name = request.form["person_name"]
    
    model["encodings"].append(encoding)
    model["names"].append(name)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    return render_template(
        "train.html",
        success="Trained Successfully"
    )

# RECOGNIZE API
@app.route("/recognize", methods=["POST"])
def recognize():

    file = request.files["image"]

    image = face_recognition.load_image_file(file)
    encodings = face_recognition.face_encodings(image)

    if len(encodings) == 0:
        return jsonify({"error": "No face found"})

    test_encoding = encodings[0]

    matches = face_recognition.compare_faces(
        model["encodings"],
        test_encoding
    )

    if True in matches:
        index = matches.index(True)

        return jsonify({
            "result": "Match Found",
            "name": model["names"][index]
        })

    return jsonify({"result": "No match found"})


if __name__ == "__main__":
    app.run(debug=True)