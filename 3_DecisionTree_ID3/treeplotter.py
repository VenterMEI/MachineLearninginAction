import matplotlib.pyplot as plt
import numpy as np
import trees

decisionNode = dict(boxstyle="sawtooth", fc="0.8")		#boxstyle 注释的边框类型，fc facecolor则为边框填充的颜色
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")						#arrowstyle 箭头类型

#添加注释的函数，将annotate这个函数进行了包装
def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt, xy=parentPt,  xycoords='axes fraction',xytext=centerPt, textcoords='axes fraction',
            va="center", ha="center", bbox=nodeType, arrowprops=arrow_args )
#annotate(s,xy,xytext) 添加注释，s为添加的字符，xy为多要标注点的位置，xytext为注释的位置，arrowprops为箭头类型，不写则只有注释没有箭头。
#xycoords为箭头尾的位置信息，一般不用设置。textcoords为箭头和注释的位置信息，一般也不用设置。	   
#va和ha是标签的对齐方式，默认是标签的左下对齐xytext。若xytext=(x,y),则va为center表明标签中心的纵坐标为y，ha为center则标签的中心的横坐标的x。
#bbox为边框，boxstyle为边框的形状，fc facecolor为边框填充色



#求myTree的宽度（叶子节点数），叶子节点数就是mytree中出现的标签的数目
def NumLeafs(myTree):
    numLeafs = 0
    firstStr = list(myTree.keys())[0]		#获得myTree的键的列表的第一个键
    secondDict = myTree[firstStr]	#获得第一个键的值，是一个嵌套的字典或一个字典
    for key in secondDict.keys():	#遍历这个有字典组成的值
        if type(secondDict[key]).__name__=='dict':		#逐个检查这个字典的每个值的类型是否为字典。注意，
            numLeafs += NumLeafs(secondDict[key])
        else:   numLeafs +=1
    	#值分为两种情况，一种是label，遇到一个label加一个;一种是字典，那就递归，看里面包含了多少label。等遍历完这个myTree，叶子节点的数目就知道了。    
    return numLeafs
#修正后myTree的宽度为叶子节点数-1
def getNumLeafs(myTree):
	return NumLeafs(myTree)-1

#求myTree的深度
def getTreeDepth(myTree):
    maxDepth = 0 
    firstStr = list(myTree.keys())[0]
    #源代码为firstStr = myTree.keys()[0]，但在python3.6中dict的items，keys操作返回的是view，要用index索引必须先将其转换成list
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:   thisDepth = 1
        		#求每个分支的深度，叶子节点深度为1，每有一个字典就说明分支一次，则深度加1。但是根节点的深度没有算入，故3层的树深度为2。
        if thisDepth > maxDepth: maxDepth = thisDepth		#树的深度为最大的分支的深度
    return maxDepth


#在父节点和子节点的箭头的中心位置添加属性值信息
def plotMidText(cntrPt, parentPt, txtString):		#cntrPt 子节点	parentPt 父节点	txtString 属性值	 
    xMid = (parentPt[0]-cntrPt[0])/2.0 + cntrPt[0]
    yMid = (parentPt[1]-cntrPt[1])/2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)	#rotation为文本旋转的角度	#text()添加纯文本信息

def plotTree(myTree, parentPt, nodeTxt):
    numLeafs = getNumLeafs(myTree)
    depth = getTreeDepth(myTree)
    firstStr = list(myTree.keys())[0]     
    cntrPt = (plotTree.xOff + numLeafs/2.0/plotTree.totalW, plotTree.yOff)	#x轴：基点xOff+偏移量(叶子节点数占totalW的比例，该比例除以2即为偏移量)
    #cntrPt = (plotTree.xOff + (1.0 + float(numLeafs))/2.0/plotTree.totalW, plotTree.yOff)	#源码
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':		#键为字典，xOff不变。但递归该字典时，其下的叶子节点一定会使xOff改变
            plotTree(secondDict[key],cntrPt,str(key))       
        else:   
            #plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW	#源码为先更改xOff值
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
            plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW		#源码中xOff为负值，故须先更改xOff的位置。经更改后现在可以反过来。
    plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD
'''
绘图的说明：
图像是(0,0)到(1,1)的正方形，xOff和yOff是每次绘图的基点。
y轴很简单，将图形划分成totalD份，基点yOff从1开始，yOff每次的偏移量设为1/totalD即可。
xOff较难理解。比如整个树有三个叶子节点，那么每个叶子节点的横坐标是多少？1/3,2/3,1？不是的，应该是1/6,1/2,5/6。否则整个图形就偏右了。
但在y轴就不存在偏上的问题，为什么？三层的树totalD=2，yOff从1开始，每次下降1/totalD=0.5，分别是1,0.5,0,图形没有往上跑。
源码中将xOff从-1/6开始，基于此先确定了偏移量，再确定子节点的计算，最后确定了xOff的值为-1/6。这不是很好理解
因此，我直接仿造y轴的方法，也将totalW自减了1，这样，xOff就可以从0开始了。
这样子做的话，公式很易于理解。但缺点是图示较源码的宽一些，美观性略差，但基本影响不大。
源码中numLeafs+1的含义是：它的这个图x轴的边界不是(0,1)，xOff是从-1/totalW开始的，即边界为(-1/totalW,1+1/totalW),所以要加1。
'''


def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = fig.add_subplot(111, frameon=False, **axprops)    #**axprops就没有了刻度，frameon=False就没有了边框
    plotTree.totalW = getNumLeafs(inTree)
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = 0; plotTree.yOff = 1.0;
    plotTree(inTree, (0.5,1.0), '')		
    #在第一次调用时，'no surfacing'属性位置与父节点(0.5,1)位置相同，故不产生箭头，且参数nodeText为空，这使得'no surfacing'的父节点形同虚设
    plt.show()
'''
def createPlot():
    fig = plt.figure(1, facecolor='white')
    fig.clf()		#clf() 清除fig之前的图像
    createPlot.ax1 = fig.add_subplot(111, frameon=False)
    plotNode('decisionNode',(0.5,0.1),(0.1,0.5),decisionNode)
    plotNode('leafNode',(0.8,0.1),(0.3,0.8),leafNode)
    plt.show()
'''

#createPlot(trees.mytree)      
