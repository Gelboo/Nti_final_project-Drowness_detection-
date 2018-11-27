from tkinter import *
import _thread
import time
import subprocess

def destroy(threadname,delay):
    time.sleep(delay)
    root.destroy()

def openn2(threadname,delay):
    time.sleep(delay)
    subprocess.call("python DrowDetectionPretrainedModel.py" , shell=True)

def GoVideo():
    try:
        _thread.start_new_thread( destroy, ("Thread-1", 0.01, ) )
        _thread.start_new_thread( openn2, ("Thread-2", 0, ) )
    except:
        print ("Error: unable to start thread")

root  = Tk()
root.bind('<Escape>', lambda e: root.quit())
root.attributes('-fullscreen', True)
root.configure(background='black')

Screen_width = root.winfo_screenwidth()
Screen_height = root.winfo_screenheight()
print(Screen_width)
Run_btn = Button(root,text='RUN',width=30,height=10,bg='blue',fg='black',font='Calibri 30',command=GoVideo)
Run_btn.place(x = 450,y = 150)
root.mainloop()
