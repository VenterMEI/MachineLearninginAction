import trees
import treeplotter

path='D:\ADA\save\python\MachineLearninginAction\Ch03\lenses.txt'
fr=open(path,'r')
dataset=[line.strip().split('\t') for line in fr.readlines()]	
#.strip(rm)        删除s字符串中开头、结尾处，出现在rm删除序列的字符
attributelist=['age','prescript','astigmatic','tearRate']
lensesTree=trees.createTree(dataset, attributelist)
treeplotter.createPlot(lensesTree)
