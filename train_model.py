from flask import Flask, request, jsonify, render_template
import face_recognition
import pickle
import os

app = Flask(__name__)

TRAIN_DIR = "train"
MODEL_PATH = "model.pkl"

# Create train folder if not exists
os.makedirs(TRAIN_DIR, exist_ok=True)

# Load or initialize model
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
else:
    model = {
        "encodings": [],
        "names": []
    }
<!-- 
# HOME PAGE (optional UI)
@app.route("/", methods=["GET"])
def home():
    return '''
    <h2>Face Training API</h2>
    <form action="/train" method="POST" enctype="multipart/form-data">
        <input type="file" name="image" required>
        <button type="submit">Train Model</button>
    </form>
    ''' -->


# TRAIN API

@app.route("/train", methods=["POST"])
def train():

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "No file selected"})

    # Save image in train folder
    image_path = os.path.join(TRAIN_DIR, file.filename)
    file.save(image_path)

    # Load image
    image = face_recognition.load_image_file(image_path)

    # Get face encoding
    encodings = face_recognition.face_encodings(image)

    if len(encodings) == 0:
        return jsonify({"error": "No face found in image"})

    encoding = encodings[0]

    # Use filename (without extension) as name
    name = os.path.splitext(file.filename)[0]

    # Add to model
    model["encodings"].append(encoding)
    model["names"].append(name)

    # Save model
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    return jsonify({
        "message": "Training successful",
        "name": name,
        "total_trained_faces": len(model["names"])
    })

# RUN SERVER
if __name__ == "__main__":
    app.run(debug=True)