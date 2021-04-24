**a basic face biometric system**  
instructions:  
choose a good quality camera for the attendance system.
In face_biometric.py:

    if default_camera:
      camera_index = 0 
    elif any_custom_camera:
      camera_index = camera_number
    video = cv2.VideoCapture(camera_index)

- create folders in "students" folder and name them according to the student names.
- put atleast 3 different photos of that student in that folder.
- e.g: let the student name be "jack"...create a folder named "jack" in "students" folder and 
  put photos of jack in the "jack" folder and then do this for all students.
- after a few minutes of scanning the class with the cam, press "q" to quit the program  
- names of the students who attended the class will be displayed in attended_students.txt
