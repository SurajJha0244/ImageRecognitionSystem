import face_recognition
import numpy as np


def recognize_face(file, model):

    image = face_recognition.load_image_file(file)

    encodings = face_recognition.face_encodings(image)

    if len(encodings) == 0:

        return {
            "status": "error",
            "message": "No face found"
        }

    test_encoding = encodings[0]

    face_distances = face_recognition.face_distance(
        model["encodings"],
        test_encoding
    )

    if len(face_distances) == 0:

        return {
            "status": "error",
            "message": "No trained faces found"
        }

    best_index = np.argmin(face_distances)

    if face_distances[best_index] < 0.5:

        name = model["names"][best_index]

        return {
            "status": "success",
            "name": name
        }

    return {
        "status": "error",
        "message": "No Match Found"
    }