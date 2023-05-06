import os
import cv2
import json
import numpy as np
from tkinter import messagebox
from tkinter import *
from PIL import Image, ImageTk





# Create the add face page
def add_face():
    # capture images for training
    def capture_images(name):
        images_dir = f"data/{name}"
        # create a new directory for the images
        os.makedirs(images_dir, exist_ok=True)

        # create a window to show the webcam feed
        window_name = "Capture Images"
        cv2.namedWindow(window_name)
        
        # capture 500 images
        cap = cv2.VideoCapture(0)
        count = 0
        while count < 500:
            ret, frame = cap.read()
            if not ret:
                continue

           # detect faces in the image
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        
            # draw a rectangle around each detected face
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, "Face Detected", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255))
                cv2.putText(frame, str(str(count)+" images captured"), (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255))
                new_img = frame[y:y+h, x:x+w]
            # show the image in the window
            cv2.imshow(window_name, frame)
            
            # save the image if a face is detected
            if len(faces) > 0:
                image_path = f"{images_dir}/{count}.jpg"
                cv2.imwrite(image_path, new_img)
                count += 1
            
                # update the number of captured images label
                numimglabel.config(text=f"Number of images captured = {count}")
        
            # exit if the 'q' key is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        messagebox.showinfo("Success", "Images are successfully captured and stored!")
        name_entry.delete(0, 'end')
        
        return count
    
    # add names to file
    def add_to_file():
        if name_entry.get() == "None":
            messagebox.showerror("Error", "Name cannot be 'None'")
            return
        elif len(name_entry.get()) == 0:
            messagebox.showerror("Error", "Name cannot be empty!")
            return
        
        name = name_entry.get()
        with open("names.txt", "r") as file:
            names = file.readlines()
        if name + "\n" not in names:
            with open("names.txt", "a") as file:
                file.write(name + "\n")
                messagebox.showinfo("Success", "Name is successfully added to database")
        
            images_dir = f"data/{name}"
            # create a new directory for the images
            os.makedirs(images_dir, exist_ok=True)
    
            capture_images(name)
            name_entry.delete(0, 'end')
            return True
        else:
            messagebox.showerror("Error", "Name is already in the database")
            return False

    # function to load the dataset
    def load_dataset():
        data_dir = "data"
        images = []
        labels = []
        label_names = []
        for name in os.listdir(data_dir):
            label_names.append(name)
            images_dir = os.path.join(data_dir, name)
            for image_file in os.listdir(images_dir):
                image_path = os.path.join(images_dir, image_file)
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                images.append(image)
                labels.append(len(label_names)-1)
        return images, labels, label_names

    # train the model
    def train_the_model():
        # load the dataset
        images, labels, label_names = load_dataset()
        # create the LBPH Face Recognizer
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        # train the recognizer
        recognizer.train(images, np.array(labels))
        # save the recognizer
        recognizer.save("face_recognizer.yml")
        # save the label names as a JSON file
        with open("label_names.json", "w") as f:
            json.dump(label_names, f)
        # display success message
        messagebox.showinfo("Success", "Face recognizer trained successfully!")
       
            

    add_face_window = Toplevel(root)
    add_face_window.title("Face Identifier")
    # create the heading in the center fo the window
    title_2 = Label(add_face_window, text="Add Face To DataBase", font=("Helvetica 30 underline"), fg="#333333", bg="#F5F5F5")
    title_2.grid(row=0, column=1, columnspan=1, sticky="ew", pady=20, padx=25)
    
    # row 1
    name = Label(add_face_window, text="Enter the name", fg="#263942", font='Helvetica 12 bold')
    name.grid(row=1, column=0,sticky="ew", pady=10, padx=5)
    name_entry = Entry(add_face_window, borderwidth=3, bg="lightgrey", font='Helvetica 11')
    name_entry.grid(row=1, column=1,sticky="ew", pady=10, padx=5)
    
    #row2
    numimglabel = Label(add_face_window, text="Number of images captured = 0", font='Helvetica 12 bold', fg="#263942")
    numimglabel.grid(row=2, column=1, columnspan=1, sticky="ew", pady=20, padx=25)
    
    # row4
    capture_button = Button(add_face_window, text="Capture Images", font=("Helvetica", 18), fg="#FFFFFF", bg="#333333", padx=20, pady=10, width = 15, command=add_to_file)
    capture_button.grid(row=4, column=0,sticky="ew", pady=10, padx=5)
    train_button = Button(add_face_window, text="Train The Model", font=("Helvetica", 18), fg="#FFFFFF", bg="#333333", padx=20, pady=10, width = 15, command=train_the_model)
    train_button.grid(row=4, column=1,sticky="ew", pady=10, padx=5)
    back_button_1 = Button(add_face_window, text="Back to Home", font=("Helvetica", 18), fg="#FFFFFF", bg="#333333", padx=20, pady=10, width = 15, command=add_face_window.destroy)
    back_button_1.grid(row=4, column=2,sticky="ew", pady=10, padx=5)

# on quit pressed
def on_close():
    if messagebox.askokcancel("Quit", "Are you sure?"):
        root.destroy()

# Create Idetify face page
def identify_face():
    # load the label names from the JSON file
    with open("label_names.json", "r") as f:
        label_names = json.load(f)
    # Create a new window
    webcam_window = Toplevel(root)
    webcam_window.title("Face Identifier")
    
    # Set the background color
    webcam_window.config(bg="#F5F5F5")
    
    # create the heading in the center fo the window
    title_1 = Label(webcam_window, text="FACE IDENTIFICATION", font=("Helvetica 30 underline"), fg="#333333", bg="#F5F5F5")
    title_1.pack(pady=3)
    
    # Create the webcam canvas in the center of the window
    webcam_canvas = Canvas(webcam_window, width=640, height=480)
    webcam_canvas.pack(side=TOP, padx=50, pady=25)

    # Create the "Hi" heading below the webcam canvas
    hi_heading = Label(webcam_window, text="Hi", font=("Helvetica", 24), fg="#333333", bg="#F5F5F5")
    hi_heading.pack(pady=10)

    # Create the "Back to Home" button in the bottom right corner
    back_button = Button(webcam_window, text="Back to Home", font=("Helvetica", 18), fg="#FFFFFF", bg="#333333", padx=20, pady=10, width = 15, command=webcam_window.destroy)
    back_button.pack(side=BOTTOM, padx=20, pady=10)

    # load the LBPH Face Recognizer
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("face_recognizer.yml")
    

    # Open the webcam and display the video feed on the canvas
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # detect faces in the image
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        # recognize each detected face
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            label, confidence = recognizer.predict(roi_gray)
            if confidence < 50:
                label_text = label_names[label]
            else:
                label_text = "Unknown"
            cv2.putText(frame,  f"Label: {label_text}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            hi_heading.config(text=f"Hi, {label_text}")
        
        # display the video stream on the canvas
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        webcam_canvas.create_image(0, 0, anchor=NW, image=imgtk)
        webcam_window.update()
    cap.release()   
    
# Create the main window
root = Tk()
root.title("Face Identifier")

# Set the background color
root.config(bg="#F5F5F5")

# Create the image on the left side
img = PhotoImage(file="1.png")
img_label = Label(root, image=img)
img_label.pack(side=LEFT, padx=50, pady=50)

# Create the heading below the image
heading = Label(root, text="HOME PAGE", font=("Helvetica 30 underline"), fg="#333333", bg="#F5F5F5")
heading.pack(pady=20)
        
# Create the three buttons on the right side
button1 = Button(root, text="Identify Face", font=("Helvetica", 18), fg="#FFFFFF", bg="#333333", padx=20, pady=10, width = 15, command=identify_face)
button2 = Button(root, text="Add New Face", font=("Helvetica", 18), fg="#FFFFFF", bg="#333333", padx=20, pady=10, width = 15, command=add_face)
button3 = Button(root, text="Quit", font=("Helvetica", 18), fg="#FFFFFF", bg="#333333", padx=20, pady=10, width = 15, command=on_close)

# Pack the buttons in a column on the right side
button1.pack(side=TOP, padx=50, pady=20)
button2.pack(side=TOP, padx=50, pady=20)
button3.pack(side=TOP, padx=50, pady=20)



# Run the main loop
root.mainloop()