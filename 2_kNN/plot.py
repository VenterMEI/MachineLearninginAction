from kNN import *
import matplotlib#将training目录下的所有文件转化为label和mat
import matplotlib.pyplot as plt


fig=plt.figure()			#创建figure类，一个figure对象可以包含一个或多个Axes对象，每个Axes都拥有一个拥有坐标系的绘图系统
ax=fig.add_subplot(1,1,1)	#add_subplot()在figure中插入子图。add_subplot(1,1,1)创建了一个只有一个子图的figure
mat,labels=file2matrix('D:\ADA\save\python\MachineLearninginAction\Ch02\datingTestSet2.txt')
print(mat.shape[0])
#ax.scatter(mat[:,1],mat[:,2])	#mat[]和mat[:]是整个矩阵，而mat[:,1]就是指的mat的整个第二列。而整个第二行是mat[1]
ax.scatter(mat[:,1],mat[:,2],s=20,c=array(labels))		#scatter():散点图方法
ax.set_title("Game & Icecream")
ax.set_xlabel("Game")		#横轴 
ax.set_ylabel("Icecream")	#纵轴
plt.show()
'''
#add_subplot(row,column,index):在figure类里插入子图
row:行  column:列
由于一个figure类可以放不止一个图，所以前两个参数的含义是figure类里每行有column个图，一共有row行。最后是一个row * column的图矩阵
index是figure中图的下标，代指这个figure的第index个图

#scatter(x, y, s=None, c=None, marker=None)：散点图方法
x和y是长度相同的一维array，分别表示点的横纵坐标
s(size)为可选项，表示点的大小
c(color)为可选项，表示点的颜色。b--blue，c--cyan，g--green，k--black m--magenta r--red w--white y--yellow
marker为点的形状，默认为'o',o表示圆形。
源码中该行为ax.scatter(Mat[:,1], Mat[:,2], 15.0*array(labels), 15.0*array(labels))
其参数等同于(Mat[:,1], Mat[:,2], s=15.0*array(labels), c=15.0*array(labels))
其中，因为各点labels值的不同，各点的大小也不同，我觉得不合适，改成统一的大小。
另外，color值应该不仅能接受'b','c'等，还可以接受数值，并根据数值自动转换成颜色。相同的数值有相同的颜色，不同的数值有不同的颜色。
所以颜色那项乘15没有意义，运行起来确实也和不乘效果相同。
'''
