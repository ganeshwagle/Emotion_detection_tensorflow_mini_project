import tensorflow as tf
from keras.preprocessing.image import img_to_array
from PIL import ImageTk
import PIL.Image
import cv2 as cv
import numpy as np
from tkinter import *
from tkinter import filedialog


model = tf.keras.models.load_model('mymodel.h5')
face_classifier = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
emotion_lables = ['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise']

def gen_frames(frame):
    gray = cv.cvtColor(frame,cv.COLOR_RGB2GRAY)
    faces = face_classifier.detectMultiScale(gray)
    
    for (x,y,w,h) in faces:
        cv.rectangle(frame,(x,y),(x+h,y+h),(0,255,255),2)
        roi_gray = gray[y:y+h,x:x+h]
        roi_gray = cv.resize(roi_gray,(48,48),interpolation=cv.INTER_AREA)
        
        if np.sum([roi_gray])!=0:
            roi = roi_gray.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi,axis=0)
            
            prediction = model.predict(roi)[0]
            label =emotion_lables[prediction.argmax()]
            label_position = (x,y+h+20)
            cv.putText(frame,label,label_position,cv.FONT_HERSHEY_SIMPLEX,0.8,(255,0,0),2)

    return frame  

root = Tk()
root.geometry('500x540')
root.title('Emotion Detection')
root.resizable(True,True)
f = Frame(root,height=540,width=500,bg='blue')
f.pack()

panel = None
choose = None
webcambutton = None
after_id = None
camera = None

def close():
    global after_id,camera
    root.after_cancel(after_id)
    camera.release()
    choose_file()
    

def webcam():
    global f,panel,webcambutton,after_id,camera
    webcambutton.destroy()
    webcambutton = Button(f,text="CLOSE",command=close,bg='dark blue',fg='white')
    webcambutton.place(x=410,y=30,height=25,width=80)
    success, frame = camera.read()
    if success:
        frame = cv.cvtColor(frame,cv.COLOR_BGR2RGB)
        img = gen_frames(frame)
        img = PIL.Image.fromarray(img)
        img = img.resize((480, 450), PIL.Image.ANTIALIAS)
        img = ImageTk.PhotoImage(img)
        panel = Label(f, image = img,bg='lightblue')
        panel.image = img
        panel.place(x=10,y=80,height=450,width=480)
        after_id = root.after(250,webcam)



def select():
    global f,panel,webcambutton
    file_path = filedialog.askopenfilename()
    img = PIL.Image.open(file_path)
    img = open_cv_image = np.array(img)
    img = gen_frames(img)
    img = PIL.Image.fromarray(img)
    img = img.resize((480, 450), PIL.Image.ANTIALIAS)  
    img = ImageTk.PhotoImage(img)
    panel = Label(f, image = img,bg='lightblue')
    panel.image = img
    panel.place(x=10,y=80,height=450,width=480)


def choose_file():
    global f,panel,choose,webcambutton,camera
    f.destroy()
    f = Frame(root,height=540,width=500,bg='blue')
    f.pack()

    camera = cv.VideoCapture(0)

    choose = Button(f,text="CHOOSE FILE",command=select,bg='dark blue',fg='white')
    choose.place(x=10,y=30,height=25,width=80)

    webcambutton = Button(f,text="USE WEBCAM",command=webcam,bg='dark blue',fg='white')
    webcambutton.place(x=410,y=30,height=25,width=80)
       
       
    panel = Label(f, text='Select the image',bg='lightblue')
    panel.place(x=10,y=80,height=450,width=480)


heading = Label(f,text="Emotion Detection",font=('times new roman',30,"bold"),bg='blue',fg='white')

name1 = Label(f,text='Ganesh Wagle',font=('times new roman',18),bg='blue',fg='white')
name2 = Label(f,text='Hawaralu Vignesh',font=('times new roman',18),bg='blue',fg='white')
name3 = Label(f,text='Karthik V Naik',font=('times new roman',18),bg='blue',fg='white')

usn1 = Label(f,text='4nm18cs058',font=('times new roman',18),bg='blue',fg='white')
usn2 = Label(f,text='4nm18cs064',font=('times new roman',18),bg='blue',fg='white')
usn3 = Label(f,text='4nm18cs071',font=('times new roman',18),bg='blue',fg='white')

heading.place(x=95,y=30)

name1.place(x=70,y=110)
name2.place(x=70,y=150)
name3.place(x=70,y=190)

usn1.place(x=325,y=110)
usn2.place(x=325,y=150)
usn3.place(x=325,y=190)


button = Button(f,text='Next',command=choose_file,bg='dark blue',fg='white')
button.place(x=225,y=280,width=90,height=30)

root.mainloop()


