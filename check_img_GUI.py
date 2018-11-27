from tkinter import *
from tkinter.filedialog import askopenfilename
import cv2
import time
import subprocess
from tkinter import messagebox
from PIL import Image,ImageTk
import numpy as np
from cascade import get_parts
from theClassify import get_status

img_name = ""
left_eye_img = None
right_eye_img = None

def load_image():
    global original_img_lbl,img_name
    Tk().withdraw() # we don't want a full GuI , so keep root window appear
    img_name = askopenfilename()
    img = cv2.imread(img_name)
    img = cv2.resize(img,(800,850))
    # cv2.imshow('img',img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img = ImageTk.PhotoImage(img)
    original_img_lbl.config(width=None,height=None)
    original_img_lbl.width=None
    original_img_lbl.height=None

    original_img_lbl.config(image=img)
    original_img_lbl.image = img
def get_component():
    global original_img_lbl,img_name,face_img_lbl,leftEye_img_lbl,rightEye_img_lbl,left_eye_img,right_eye_img
    img = cv2.imread(img_name)
    img,faces,eyes = get_parts(img)
    img = cv2.resize(img,(800,850))
    img = Image.fromarray(img)
    img = ImageTk.PhotoImage(img)
    original_img_lbl.config(image=img)
    original_img_lbl.image=img
    if faces == []:
        messagebox.showinfo(title="Warning",message="Image contain No faces")
    if eyes == []:
        messagebox.showinfo(title="Warning",message="Can't Detect any Eye")
    num_faces = len(faces)
    num_eyes = len(eyes)
    face = faces[0]
    face = cv2.resize(face,(645,430))
    face = Image.fromarray(face)
    face = ImageTk.PhotoImage(face)
    face_img_lbl.config(image=face)
    face_img_lbl.image = face

    left_eye = eyes[0]
    left_eye_img = left_eye
    left_eye = cv2.resize(left_eye,(480,260))
    left_eye = Image.fromarray(left_eye)
    left_eye = ImageTk.PhotoImage(left_eye)
    leftEye_img_lbl.config(image=left_eye)
    leftEye_img_lbl.image=left_eye

    right_eye = eyes[1]
    right_eye_img = right_eye
    right_eye = cv2.resize(right_eye,(480,260))
    right_eye = Image.fromarray(right_eye)
    right_eye = ImageTk.PhotoImage(right_eye)
    rightEye_img_lbl.config(image=right_eye)
    rightEye_img_lbl.image=right_eye
def check_status():
    global left_eye_img,right_eye_img
    if left_eye_img is None or right_eye_img is None:
        messagebox.showinfo(title="warning",message="There is no eye Component \n Please check")
    res_left = get_status(left_eye_img,'left_eye')
    res_right = get_status(right_eye_img,'right_eye')
    print(res_left,res_right)

root = Tk()

root.bind('<Escape>', lambda e: root.quit())
root.attributes('-fullscreen', True)
root.configure(background='blue')


load_btn = Button(root,text="Load an Image",width=12,height=2,fg='black',bg='yellow',font='Calibri 20',command=load_image)
get_image_component_btn = Button(root,text="Get Image Component",width=18,height=2,fg='black',bg='yellow',font='Calibri 20',command=get_component)
original_img_PH = Label(root,text='Img',bg='white',width=100,height=50)
original_img_lbl = Label(root,text='',bg='white')

face_img_PH = Label(root,text='Face \n Img',bg='white',width=80,height=25)
leftEye_img_PH = Label(root,text='Left \n Eye',bg='white',width=60,height=15)
rightEye_img_PH = Label(root,text='Right \n Eye',bg='white',width=60,height=15)

face_img_lbl = Label(root,bg='white')
leftEye_img_lbl = Label(root,bg='white')
rightEye_img_lbl = Label(root,bg='white')

check_status_btn = Button(root,text="Check The person Status",width=20,height=2,fg='black',bg='yellow',font='Calibri 20',command=check_status)
Res_lbl = Label(root,text="Res:",font='Calibri 20',bg='blue',fg='white')


load_btn.place(x=100,y=50)
get_image_component_btn.place(x=450,y=50)
original_img_lbl.place(x=1100,y=50)
original_img_PH.place(x=1100,y=50)
face_img_PH.place(x=150,y=240)
leftEye_img_PH.place(x=20,y=700)
rightEye_img_PH.place(x=580,y=700)
face_img_lbl.place(x=150,y=240)
leftEye_img_lbl.place(x=20,y=700)
rightEye_img_lbl.place(x=580,y=700)
check_status_btn.place(x=1200,y=960)
Res_lbl.place(x=1700,y=940)
root.mainloop()
