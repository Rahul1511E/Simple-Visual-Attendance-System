import cv2
import os
import pyttsx3
import pandas as pd
import tkinter as tk
from datetime import datetime
from PIL import Image, ImageTk
from tkinter import messagebox

def show_main_page():
    for widget in root.winfo_children():
        widget.destroy()

    frame1 = tk.Frame(root)
    frame1.pack(side=tk.TOP, pady=5)

    capture_button = tk.Button(frame1, text="Capture Images", command=show_capture_page)
    capture_button.pack(side=tk.LEFT)

    recognizer_button = tk.Button(frame1, text="Start Recognizer", command=recognizer)
    recognizer_button.pack(side=tk.LEFT)

def capture_images(name):
    if not name:
        return

    # initialize the camera
    cap = cv2.VideoCapture(0)
    classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    # directory for saving images
    dirName = "./" + name
    os.makedirs(dirName, exist_ok=True)

    count = 200  # number of images to capture
    counter = 0  # counter for captured images
    while counter < count:
        _, image = cap.read()
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = classifier.detectMultiScale(gray_image, 1.1, 5)
        for(x, y, w, h) in faces:
            roi_gray = gray_image[y:y + h, x:x + w]
            cv2.imwrite(dirName + "/" + name + str(counter) + ".jpg", roi_gray)
            counter += 1
    cap.release()

    tk.messagebox.showinfo(title="Info", message="Completed capturing")
    show_main_page()


def show_capture_page():
    for widget in root.winfo_children():
        widget.destroy()

    frame1 = tk.Frame(root)
    frame1.pack(side=tk.TOP, pady=5)

    tk.Label(frame1, text="Enter your name:").pack(side=tk.LEFT)
    name_entry = tk.Entry(frame1)
    name_entry.pack(side=tk.LEFT)

    tk.Button(frame1, text="Submit", command=lambda: capture_images(name_entry.get())).pack(side=tk.LEFT)


def recognizer():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('trainer/trainer.yml')
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Text-to-speech engine
    engine = pyttsx3.init()

    id = 0
    names = ['Unknown']  # the list of available persons

    # start real-time video capture
    cam = cv2.VideoCapture(0)
    cam.set(3, 640)  # set video width
    cam.set(4, 480)  # set video height

    # define minimal window size to be recognized as a face
    minW = 0.1 * cam.get(3)
    minH = 0.1 * cam.get(4)

    while True:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(int(minW), int(minH)))
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            id, confidence = recognizer.predict(gray[y:y + h, x:x + w])
            if(confidence < 100):
                id = names[id]
                confidence = "  {0}%".format(round(100 - confidence))
            else:
                id = "unknown"
                confidence = "  {0}%".format(round(100 - confidence))
                engine.say("Unknown person, please sign in register")
                engine.runAndWait()
            cv2.putText(img, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
            cv2.putText(img, str(confidence), (x + 5, y + h - 5), font, 1, (255, 255, 0), 2)
            if id != 'Unknown':
                engine.say("Welcome " + id)
                engine.runAndWait()
                now = datetime.now()
                dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
                if not os.path.isfile('Attendance.csv'):
                    df = pd.DataFrame([[id, dt_string]], columns=["Name", "Time"])
                    df.to_csv('Attendance.csv', index=False)
                else:
                    df = pd.read_csv('Attendance.csv')
                    new_row = pd.DataFrame([[id, dt_string]], columns=["Name", "Time"])
                    df = pd.concat([df, new_row])
            df.to_csv('Attendance.csv', index=False)

    cam.release()
    cv2.destroyAllWindows()

root = tk.Tk()
show_main_page()
root.mainloop()
