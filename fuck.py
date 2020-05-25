"""
@Author : 行初心
@Date   : 18-10-1
@Blog   : www.cnblogs.com/xingchuxin
@Gitee  : gitee.com/zhichengjiu
"""
from tkinter import *


def main():
    root = Tk()  # 注意Tk的大小写

    photo = PhotoImage(file='fuck.png')
    img_label = Label(root, imag=photo)
    img_label.pack()

    mainloop()


if __name__ == '__main__':
    main()