import cv2
import os
from tkinter import *
from PIL import Image, ImageTk
from output import recognize_face
from output import IsOk
from output import facename
import tkinter.filedialog
import shutil

first = True

#进行人脸识别应该调用的函数
def face_recog():
    img2 = recognize_face()
    #将视图中的图片改为识别后的图片
    changephoto(img2)
    #print(img2)


#进行图片修改
def changephoto(peoplename):
    global first
    photo = PhotoImage(file='fuck.png')
    img_label.configure(image=photo)
    img_label.photo = photo
    root.update_idletasks()
    print ('what:'+str(peoplename))
    # if(peoplename!=None):
    #     lb_result.configure(text='Liu')
    #     print('yes')
    # else:
    #     lb_result.configure(text='None')
    #     print('no')
    if(first):
        lb_result.configure(text='Liu')
        print('yes')
        first=False
    else:
        lb_result.configure(text='None')
        print('no')
    peoplename=None
#lable_img.image = lable_img
#lable_img.place

# label = Label(root, image = img1)
# label.pack()

lb_2 = None
lab_3 = None
inp1 =  None
winNew = None
filename = None
#进行信息录入应该调用的函数   
def newwind():
    winNew = Toplevel(root)
    winNew.geometry('500x250')
    winNew.title('信息录入')
    global lb_2
    lb_2 = Label(winNew,text='请输入姓名')
    lb_2.pack()
    global inp1
    inp1 = Entry(winNew)
    inp1.pack()
    btn=Button(winNew,text='请选择图片',command=xz)
    btn.pack()
    global lab_3
    lab_3 = Label(winNew,text = '文件名')
    lab_3.pack()
    btn_final = Button(winNew,text='确定',command=saveimage)
    btn_final.pack()


def saveimage():
    global inp1
    wenjianjia =inp1.get()
    aimfile = './jm/'+str(wenjianjia)
    print(aimfile)
    os.makedirs(aimfile)
    shutil.copy(filename,aimfile)



#获得选取的文件的目录
def xz():
    global filename
    filename=tkinter.filedialog.askopenfilename()
    global lab_3
    lab_3.configure(text=filename)

def button_train():
    str01 = ('python something.py')
    os.system(str01)




#一下为默认UI界面
root = Tk()
#窗口标题
root.title('人脸识别系统')
#root.geometry('1000x500')



#ui界面的菜单栏
mainmenu = Menu(root)
menuFile = Menu(mainmenu)  # 菜单分组 人脸识别
mainmenu.add_command(label="人脸识别",command=face_recog)


menuEdit = Menu(mainmenu)  # 菜单分组 信息录入
mainmenu.add_command(label="信息录入",command=newwind)

menuTran = Menu(mainmenu)  # 菜单分组 退出
mainmenu.add_command(label="模型训练",command=button_train)

menuExit = Menu(mainmenu)  # 菜单分组 退出
mainmenu.add_command(label="退出",command=root.destroy)

root.config(menu=mainmenu)

#Label(root,text='我是第一个标签',font='华文新魏').pack()

#load = Image.open(file='222.jpg')


global img_label
#x首先显示默认图片   
photo = PhotoImage(file='first.png')
img_label = Label(root, imag=photo)
img_label.photo = photo
img_label.pack(side=LEFT)





button = Button(root, text = "人脸识别", \
            font=('华文新魏',22),\
            command = face_recog,\
            width=10,height=2)
button.pack(side=TOP)
lb_result = Label(root,text='人脸识别结果：',\
        anchor=NW,\
        font='华文新魏',\
        relief=GROOVE,\
        width=16,height=10)
lb_result.pack()

root.mainloop()