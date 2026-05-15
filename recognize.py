import face_recognition
import pickle
import json
import os
import webb

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Load database
with open("database.json", "r") as f:
    database = json.load(f)


# USER INPUT IMAGE FILE

file_name = input("Enter image name: ")
image_path = os.path.join("test", file_name)
 
# Check file exists
if not os.path.exists(image_path):
    print("File not found!")
    exit()

# Load image
image = face_recognition.load_image_file(image_path)

# Encode face
encodings = face_recognition.face_encodings(image)

if len(encodings) == 0:
    print("No face found in image")
    exit()

test_encoding = encodings[0]

# Compare with trained faces
matches = face_recognition.compare_faces(model["encodings"], test_encoding)

name = "Unknown"

if True in matches:
    index = matches.index(True)
    name = model["names"][index]

    print("\nFace Recognized:", name)

    # Normalize key
    name = name.lower().strip()

    # Fetch from database
    if name in database:
        details = database[name]

        print("\n--- Person Details ---")
        print("Name:", details.get("name"))
        print("Age:", details.get("age"))
        print("Role:", details.get("role"))
        print("Email:", details.get("email"))
        print("Phone:",details.get("phone"))
        print("City:",details.get("city"))
    else:
        print("No details found in database.json")

else:
    print("No match found")