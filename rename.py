import os

# 需要改变的图像文件的路径，我放于桌面了
path ='C:\\Users\\jomain\\Desktop\\deeplearning\\lmy\\fuck\\Yale\\6'
# 改变后存放图片的文件夹路径，我也放于桌面了
path1 = 'C:\\Users\\jomain\\Desktop\\deeplearning\\lmy\\jm'
filelist = os.listdir(path)

j = 155

for i in filelist:
    # 判断该路径下的文件是否为图片
    if i.endswith('.bmp'):#png可以改为jpg
        # 打开图片
        src = os.path.join(os.path.abspath(path), i)
        # 重命名
        dst = os.path.join(os.path.abspath(path1), 's' + format(str(j), '0>s')+ '.bmp')#0>s的意思是 图片的名称没有0，例如1_label.png，
                                                                                             #   如果改为0>3s，则结果为001_label.png
        # 执行操作
        os.rename(src, dst)
        j += 1