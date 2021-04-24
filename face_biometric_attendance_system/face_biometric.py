import face_recognition
import os
import cv2
import pickle
import datetime

save = False
load = False
resize = True
scale_percent = 40

print('')
print(f'Model Status : Save:{save} | Load:{load}')

KNOWN_FACES_DIR = 'students'
TOLERANCE = 0.6
FRAME_THICKNESS = 4
FONT_THICKNESS = 2
MODEL = 'hog'  # hog or cnn
video = cv2.VideoCapture(0)


def name_to_color(name):
    color = [(ord(c.lower())-97)*8 for c in name[:3]]
    return color


if load:
    print('')
    print("Loading students' faces and students' names...")

    with open('models/known_faces_model.pkl', 'rb') as f:
        known_faces = pickle.load(f)
    with open('models/known_names_model.pkl', 'rb') as f:
        known_names = pickle.load(f)

    print('')
    print("Done loading students' and students' names")
else:
    print('')
    print('Did not load the models')
    print('')
    print("Processing students' faces...")

    known_faces = []
    known_names = []
    for name in os.listdir(KNOWN_FACES_DIR):
        for filename in os.listdir(f'{KNOWN_FACES_DIR}/{name}'):
            image = face_recognition.load_image_file(
                f'{KNOWN_FACES_DIR}/{name}/{filename}')
            encodings = face_recognition.face_encodings(image)[0]
            known_faces.append(encodings)
            known_names.append(name)

    print('')
    print("Done processing students' faces!!")

if save:
    print('')
    print("Saving students' faces and students' names...")

    with open('models/known_faces_model.pkl', 'wb') as f:
        pickle.dump(known_faces, f)
    with open('models/known_names_model.pkl', 'wb') as f:
        pickle.dump(known_names, f)

    print('')
    print('Saved succesfully!!')
else:
    print('')
    print('Did not save the models')

faces = []
x = datetime.datetime.now()
faces.append(f"Date: {x}")
faces.append('')
faces.append("Attended Students:")
faces.append('')

print('')
print('Loading the camera feed...')

while True:
    _, image = video.read()

    locations = face_recognition.face_locations(image, model=MODEL)
    encodings = face_recognition.face_encodings(image, locations)

    for face_encoding, face_location in zip(encodings, locations):
        results = face_recognition.compare_faces(
            known_faces, face_encoding, TOLERANCE)
        match = None
        if True in results:
            match = known_names[results.index(True)]
            # print(f'Found match : {match}')

            if match not in faces:
                faces.append(match)

            top_left = (int(face_location[3]), int(face_encoding[0]))
            bottom_right = (int(face_location[1]), int(face_location[2]))
            color = name_to_color(match)
            cv2.rectangle(image, top_left, bottom_right,
                          color, FRAME_THICKNESS)

            top_left = (int(face_location[3]), int(face_location[2]))
            bottom_right = (int(face_location[1]), int(face_location[2]) + 22)
            cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)
            cv2.putText(image, match, (int(face_location[3]) + 2, int(
                face_location[2]) + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), FONT_THICKNESS)

    cv2.imshow("Camera Feed", image)
    key = cv2.waitKey(30)
    if key == ord('q'):
        break

# print('')
# print(f"Attented Students: {faces}")

with open('attended_students.txt', 'w') as f:
    for name in faces:
        f.write("%s\n" % name)
