import face_recognition
import pickle
import os

TRAIN_DIR = "train"
MODEL_PATH = "model.pkl"

os.makedirs(TRAIN_DIR, exist_ok=True)


# LOAD MODEL
def load_model():

    if os.path.exists(MODEL_PATH):

        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)

    else:

        model = {
            "encodings": [],
            "names": []
        }

    return model


# SAVE MODEL
def save_model(model):

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)


# RETRAIN MODEL
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

    model = {
        "encodings": known_encodings,
        "names": known_names
    }

    save_model(model)

    return model


# TRAIN FACE
def train_face(name, file, model):

    if file.filename == "":
        return {
            "status": "error",
            "message": "No file selected"
        }

    extension = os.path.splitext(file.filename)[1]

    unique_filename = f"{name}_{len(model['names'])}{extension}"

    image_path = os.path.join(TRAIN_DIR, unique_filename)

    file.save(image_path)

    image = face_recognition.load_image_file(image_path)

    encodings = face_recognition.face_encodings(image)

    if len(encodings) == 0:

        os.remove(image_path)

        return {
            "status": "error",
            "message": "No face found in image"
        }

    encoding = encodings[0]

    # DUPLICATE CHECK
    if len(model["encodings"]) > 0:

        distances = face_recognition.face_distance(
            model["encodings"],
            encoding
        )

        best_index = distances.argmin()

        if distances[best_index] < 0.5:

            existing_name = model["names"][best_index]

            return {
                "status": "duplicate",
                "old_name": existing_name,
                "new_name": name
            }

    model["encodings"].append(encoding)
    model["names"].append(name)

    save_model(model)

    return {
        "status": "success",
        "name": name
    }


# UPDATE FACE NAME
def update_face_name(old_name, new_name, model):

    updated = False

    for i in range(len(model["names"])):

        if model["names"][i] == old_name:

            model["names"][i] = new_name
            updated = True

    if updated:

        save_model(model)

        return True

    return False