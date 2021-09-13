import numpy as np 
import pandas as pd 

import matplotlib.pyplot as plt

import pickle

import tkinter as tk
from PIL import ImageTk,Image,ImageDraw
import cv2

# Managing Warnings 
import warnings
warnings.filterwarnings('ignore')

filename = 'clf.sav'
output_path='C:/Users/MASNA.CO/Desktop\Data Science/Arabic Handwritten Digits Recognition/output/'
clf = pickle.load(open(output_path+filename, 'rb'))
print('mission done!')

def event_function(event):

    x=event.x       #x coordinate of mouse pointer
    y=event.y       #y coordinate of mouse pointer

    x1=x-20
    y1=y-20

    x2=x+20
    y2=y+20
    
    canvas.create_oval((x1,y1,x2,y2),fill='black')
    img_draw.ellipse((x1,y1,x2,y2),fill='white')

def save():

    global count
    
    img_array=np.array(img)
    img_array=cv2.resize(img_array,(28,28))

    path=os.path.join('./output/test_Data',str(count)+'.jpg')
    
    cv2.imwrite(path,img_array)

    count=count+1

def clear():

    global img,img_draw

    canvas.delete('all')
    img=Image.new('RGB',(width,height),(0,0,0))
    img_draw=ImageDraw.Draw(img)
    label_result.config(text='Predicted label: ---')

def predict():
    global img

    img_array=np.array(img) #converting to numpy array
    img_array=cv2.cvtColor(img_array,cv2.COLOR_BGR2GRAY) #converting into a gray image
    img_array1=cv2.resize(img_array.T,(28,28)) #resizing into 28x28    # T to be compatible with the train data

    img_array=np.reshape(img_array1,(1,784))  
    img_array=img_array/255.0

    result=clf.predict(img_array)

    label_result.config(text='Predicted label:  '+str(result[0]),)
    plt.imshow(img_array1.T, cmap='gray',)
    plt.show()

try:
    os.mkdir('./output/test_Data')
except:
    print('Path Already Exists')

    
width,height=500,500

root=tk.Tk()

root.geometry("+400+50")

font='Times 20 bold'
count=0
    
frame = tk.Frame(root, bg='#80c1ff', bd=3)
frame.grid(row=0,column=0,columnspan=4)

label_result=tk.Label(frame,text='Predicted label: ---',bg='#80c1ff',font=font)
label_result.grid(row=0,column=0,columnspan=4)

canvas=tk.Canvas(root,width=width,height=height,bg='#CCE5FF',bd=15)
canvas.grid(row=1,column=0,columnspan=4)

canvas.bind('<B1-Motion>',event_function)
img=Image.new('RGB',(width,height),(0,0,0))
img_draw=ImageDraw.Draw(img)


button_predict=tk.Button(root,text='Predict',bg='#4C9900',fg='black',font=font,command=predict)  #4C9900  : green
button_predict.grid(row=2,column=0)

button_clear=tk.Button(root,text='Clear',bg='#C0C0C0',fg='black',font=font,command=clear)  #C0C0C0  :gray
button_clear.grid(row=2,column=1)

button_save=tk.Button(root,text='Save',bg='#66B2FF',fg='black',font=font,command=save)   #66B2FF : blue
button_save.grid(row=2,column=2)

button_exit=tk.Button(root,text='Exit',bg='#FF66FF',fg='black',font=font,command=root.destroy)  #FF66FF  :purple
button_exit.grid(row=2,column=3)

root.mainloop()