from tkinter import *
from tkinter.filedialog import askopenfilename
import cv2
import time
import subprocess
from tkinter import messagebox
from PIL import Image,ImageTk
import numpy as np
from cascade import get_parts
from detect_faces_eyes_dlib import get_components_dlib
from theClassify import get_status
import threading

img_name = ""
left_eye_img = None
right_eye_img = None
v = None
which_algorithm_name = ""
def pause():
    global v,original_img_lbl,face_img_lbl,leftEye_img_lbl,rightEye_img_lbl
    v.release()
    leftEye_img_lbl.config(image=None)
    leftEye_img_lbl.image=None

    rightEye_img_lbl.config(image=None)
    rightEye_img_lbl.image=None

    face_img_lbl.config(image=None)
    face_img_lbl.image=None

    original_img_lbl.config(image=None)
    original_img_lbl.image = None
stopEvent = threading.Event()
def start():
    global which_algorithm_name
    which_algorithm_name = which_algorithm.get()
    thread = threading.Thread(target=videoLoop, args=())
    thread.start()
def videoLoop():
    global stopEvent,which_algorithm_name,v,original_img_lbl,img_name,face_img_lbl,leftEye_img_lbl,rightEye_img_lbl,Res_value
    v = cv2.VideoCapture(0)
    while True:
        r,O_img = v.read()
        if not r:
            break
        R_img = cv2.resize(O_img,(800,850))
        # cv2.imshow('img',img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        img = cv2.cvtColor(R_img,cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = ImageTk.PhotoImage(img)
        original_img_lbl.config(width=None,height=None)
        original_img_lbl.width=None
        original_img_lbl.height=None

        original_img_lbl.config(image=img)
        original_img_lbl.image = img
        if which_algorithm_name == 'Haarcascade':
            img,faces,eyes = get_parts(O_img)
        else:
            img,faces,eyes = get_components_dlib(O_img)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img = cv2.resize(img,(800,850))
        img = Image.fromarray(img)
        img = ImageTk.PhotoImage(img)
        original_img_lbl.config(image=img)
        original_img_lbl.image=img
        num_faces = len(faces)
        num_eyes = len(eyes)

        if num_faces < 1:
            continue
        face = faces[0]
        face = cv2.resize(face,(645,430))
        face = cv2.cvtColor(face,cv2.COLOR_BGR2RGB)
        face = Image.fromarray(face)
        face = ImageTk.PhotoImage(face)
        face_img_lbl.config(image=face)
        face_img_lbl.image = face
        if num_eyes < 2:
            continue
        left_eye = eyes[0]
        left_eye_img = cv2.imread('Ally_Sheedy_0001_L.jpg')
        left_eye = cv2.resize(left_eye,(480,260))
        left_eye = cv2.cvtColor(left_eye,cv2.COLOR_BGR2RGB)
        left_eye = Image.fromarray(left_eye)
        left_eye = ImageTk.PhotoImage(left_eye)
        leftEye_img_lbl.config(image=left_eye)
        leftEye_img_lbl.image=left_eye
        print(left_eye_img.shape)
        # left_res = get_status(left_eye_img,'left_eye')


        right_eye = eyes[1]
        right_eye_img = cv2.imread('Ally_Sheedy_0001_L.jpg')
        right_eye = cv2.resize(right_eye,(480,260))
        right_eye = cv2.cvtColor(right_eye,cv2.COLOR_BGR2RGB)
        right_eye = Image.fromarray(right_eye)
        right_eye = ImageTk.PhotoImage(right_eye)
        rightEye_img_lbl.config(image=right_eye)
        rightEye_img_lbl.image=right_eye
        # right_res = get_status(right_eye_img,'right_eye')
        # Res_value.config(text=res_left+" \n "+res_right)
        # Res_value.text = res_left+" \n "#+res_right

root = Tk()

root.bind('<Escape>', lambda e: root.quit())
root.attributes('-fullscreen', True)
root.configure(background='green')


start_btn = Button(root,text="START",width=20,height=4,fg='black',bg='yellow',font='Calibri 20',command=start)
pause_btn = Button(root,text="PAUSE",width=20,height=4,fg='black',bg='yellow',font='Calibri 20',command=pause)
original_img_PH = Label(root,text='Img',bg='white',width=100,height=50)
original_img_lbl = Label(root,text='',bg='white')

face_img_PH = Label(root,text='Face \n Img',bg='white',width=80,height=25)
leftEye_img_PH = Label(root,text='Left \n Eye',bg='white',width=60,height=15)
rightEye_img_PH = Label(root,text='Right \n Eye',bg='white',width=60,height=15)

face_img_lbl = Label(root,bg='white')
leftEye_img_lbl = Label(root,bg='white')
rightEye_img_lbl = Label(root,bg='white')

Res_lbl = Label(root,text="Res:",font='Calibri 20',bg='green',fg='white')
Res_value = Label(root,text='',font='Calibri 20',bg='green',fg='red')

which_algorithm = StringVar(root)
which_algorithm.set("Haarcascade")
choose_Algo = OptionMenu(root,which_algorithm,"Dlib","Haarcascade")
choose_Algo.config(fg='black')
choose_Algo.config(bg='brown')
choose_Algo.config(width=10)
choose_Algo.config(font='Calibri 20')
choose_Algo.place(x=860,y=120)

start_btn.place(x=100,y=50)
pause_btn.place(x=480,y=50)
original_img_lbl.place(x=1100,y=50)
original_img_PH.place(x=1100,y=50)
face_img_PH.place(x=150,y=240)
leftEye_img_PH.place(x=20,y=700)
rightEye_img_PH.place(x=580,y=700)
face_img_lbl.place(x=150,y=240)
leftEye_img_lbl.place(x=20,y=700)
rightEye_img_lbl.place(x=580,y=700)
Res_lbl.place(x=1700,y=940)
Res_value.place(x=1700,y=980)
root.mainloop()
