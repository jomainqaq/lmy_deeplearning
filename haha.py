# python批量更换后缀名
import os
import sys
os.chdir(r'C:/Users/jomain\Desktop/deeplearning/lmy/me')

# 列出当前目录下所有的文件
files = os.listdir('./')
print('files',files)

for fileName in files:
	portion = os.path.splitext(fileName)
	# 如果后缀是.dat 
	if portion[1] == ".gif":
		#把原文件后缀名改为 txt
		newName = portion[0] + ".jpg" 
		os.rename(fileName, newName)