from __future__ import print_function
import requests
import jsonpickle
import os
from tkinter import messagebox
from PIL import ImageEnhance as Enhance
from tkinter import *
from PIL import ImageTk,Image
from tkinter import font as tkFont
import cv2
import pyttsx3
from keras.preprocessing.image import load_img,img_to_array
from time import sleep
from tkinter import filedialog

import numpy as np
addr = 'http://localhost:5000'
test_url = addr + '/api/classification'
# prepare headers for http request
content_type = 'image/jpg'
tk=Tk()
headers = {'content-type': content_type}
def take_image():
    key = cv2.waitKey(1)
    webcam = cv2.VideoCapture(0)
    sleep(2)
    img=[]
    while True:
        try:
            check, frame = webcam.read()
            cv2.imshow("Capturing", frame)
            key = cv2.waitKey(1)
            if key == ord('s'):
                cv2.imwrite(filename='saved_img.jpg', img=frame)
                webcam.release()
                cv2.destroyAllWindows()
                break
            elif key == ord('q'):
                webcam.release()
                cv2.destroyAllWindows()
                break
        except(KeyboardInterrupt):
            print("Turning off camera.")
            webcam.release()
            print("Camera off.")
            print("Program ended.")
            cv2.destroyAllWindows()
            break
    clssification("saved_img.jpg")
def clssification(image):
    img =load_img(image)
    image = Enhance.Sharpness(img)
    out = image.enhance(2)
    image=img_to_array(out)

    img = np.uint8(img)
    img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
    cv2.imshow("Currency image", img)
    cv2.waitKey(1)


    ed, img_encoded = cv2.imencode('.jpg', image)
    # send http request with image and receive response
    response = requests.post(test_url, data=img_encoded.tostring(), headers=headers)
    # decode response
    response_pickled = jsonpickle.decode(response.text)
    out=str(response_pickled['message'])
    # init function to get an engine instance for the speech synthesis
    messagebox.showinfo("Value",str(out)+' LE' )


    engine = pyttsx3.init()
    # say method on the engine that passing input text to be spoken
    engine.say('this is ' + str(out)+'egyptian pound')
    # run and wait method, it processes the voice commands.
    engine.runAndWait()
    #os.remove('saved_img.jpg')
    cv2.destroyAllWindows()


def chooseImage():
    tk.sourceFile = filedialog.askopenfilename(parent=tk, initialdir= "/", title='Please select a directory')
    clssification(tk.sourceFile)
def finish():
    tk.destroy()
tk.title('Money Interpreter')
tk.geometry('1300x700')
canvas=Canvas(tk,width=1300,height=866)
image=ImageTk.PhotoImage(Image.open("back.jpg"))
canvas.create_image(0,0,anchor=NW,image=image)
canvas.pack()
myFont =tkFont.Font(family='Helvetica', size=20, weight='bold')
takeIm=Button(tk,text='Take Image',width=10,height=2,command=take_image)
takeIm['font'] = myFont
takeIm.place(x=900,y=433)
finish=Button(tk,text='finish',width=10,height=2,command=finish)
finish['font']=myFont
finish.place(x=100,y=433)
choosImage =Button(tk, text = "Choose Image", width = 12, height = 2, command = chooseImage)
choosImage['font']=myFont
choosImage.place(x=500,y=433)
tk.mainloop()