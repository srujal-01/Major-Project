import face_recognition
import pickle
import cv2
import os

print("[INFO] Starting to process faces...")
# Get paths of all image files in the dataset
imagePaths = []
for root, dirs, files in os.walk("dataset"):
    for file in files:
        if file.endswith(('.png', '.jpg', '.jpeg')):
            imagePaths.append(os.path.join(root, file))

knownEncodings = []
knownNames = []

# Loop over the image paths
for (i, imagePath) in enumerate(imagePaths):
    # Extract the person name from the image path
    print(f"[INFO] Processing image {i + 1}/{len(imagePaths)}: {imagePath}")
    # The name is the directory name
    name = imagePath.split(os.path.sep)[-2]

    # Load the image and convert it from BGR (OpenCV default) to RGB
    image = cv2.imread(imagePath)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect the (x, y)-coordinates of the bounding boxes
    # corresponding to each face in the input image
    boxes = face_recognition.face_locations(rgb, model="hog")

    # Compute the facial embedding for the face
    encodings = face_recognition.face_encodings(rgb, boxes)

    # Loop over the encodings
    for encoding in encodings:
        # Add each encoding + name to our set of known names and encodings
        knownEncodings.append(encoding)
        knownNames.append(name)

# Dump the facial encodings + names to disk
print("[INFO] Serializing encodings...")
data = {"encodings": knownEncodings, "names": knownNames}
with open("encodings.pickle", "wb") as f:
    f.write(pickle.dumps(data))

print("[INFO] Encodings saved to encodings.pickle")