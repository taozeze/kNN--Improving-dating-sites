from numpy import *
import matplotlib as mpl
import matplotlib.pyplot as plt
import operator

def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize,1)) - dataSet  #统一矩阵，实现加减
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)  #进行累加，axis=0是按列，axis=1是按行
    distances = sqDistances**0.5  #开根号
    sortedDistIndicies = distances.argsort()  #按升序进行排序，返回原下标
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1  #get是字典中的方法，前面是要获得的值，后面是若该值不存在时的默认值
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def file2matrix(filename):  #获取数据
    f = open(filename)
    arrayOLines = f.readlines()
    numberOfLines = len(arrayOLines)
    returnMat = zeros((numberOfLines,3),dtype=float)
    #zeros(shape, dtype, order),创建一个shape大小的全为0矩阵，dtype是数据类型，默认为float，order表示在内存中排列的方式（以C语言或Fortran语言方式排列），默认为C语言排列
    classLabelVector = []
    rowIndex = 0
    for line in arrayOLines:
        line = line.strip()
        listFormLine = line.split('\t')
        returnMat[rowIndex,:] = listFormLine[0:3]
        classLabelVector.append(int(listFormLine[-1]))
        rowIndex += 1
    return returnMat, classLabelVector


def autoNorm(dataSet): #归一化数值
    minVals = dataSet.min(0)  #0表示每列的最小值，1表示每行的最小值，以一维矩阵形式返回
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m,1))
    normDataSet = normDataSet/tile(ranges, (m,1))
    return normDataSet, ranges, minVals


def datingClassTest(datingDataMat, datingLabels):  #测试正确率
    hoRatio = 0.1
    m = datingDataMat.shape[0]
    numTestVecs = int(hoRatio*m)
    numError = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(datingDataMat[i,:], datingDataMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        print('The classifier came back with: %d, the real answer is: %d.' %(classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]):
            numError += 1
    print('错误率为 %f' %(numError/float(numTestVecs)))


def classifyPerson(datingDataMat, datingLabels, ranges, minVals):
    result = ['not at all', 'in small doses', 'in large doses']
    print('请输入相应信息：')
    percentTats = float(input('percentage of time spent playing video games?'))
    ffMiles = float(input('frequent flier miles earned per year?'))
    iceCream = float(input('liters of ice cream consumed per year?'))
    inArr = array([ffMiles, percentTats, iceCream])
    classifyResult = classify0((inArr-minVals)/ranges, datingDataMat, datingLabels, 3)
    print('You will probably like this person: ', result[classifyResult-1])


if __name__ == "__main__":
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    datingDataMat, ranges, minVals = autoNorm(datingDataMat) #归一化数值
    datingClassTest(datingDataMat, datingLabels)
    classifyPerson(datingDataMat, datingLabels, ranges, minVals)
    fig = plt.figure()  #图
    plt.title('散点分析图')
    mpl.rcParams['font.sans-serif'] = ['KaiTi']
    mpl.rcParams['font.serif'] = ['KaiTi']
    plt.xlabel('每年获取的飞行常客里程数')
    plt.ylabel('玩视频游戏所耗时间百分比')
    '''
    matplotlib.pyplot.ylabel(s, *args, **kwargs)

    override = {
       'fontsize'            : 'small',
       'verticalalignment'   : 'center',
       'horizontalalignment' : 'right',
       'rotation'='vertical' : }
    '''

    type1_x = []
    type1_y = []
    type2_x = []
    type2_y = []
    type3_x = []
    type3_y = []
    ax = fig.add_subplot(111) #将图分成1行1列，当前坐标系位于第1块处（这里总共也就1块）

    index = 0
    for label in datingLabels:
        if label == 1:
            type1_x.append(datingDataMat[index][0])
            type1_y.append(datingDataMat[index][1])
        elif label == 2:
            type2_x.append(datingDataMat[index][0])
            type2_y.append(datingDataMat[index][1])
        elif label == 3:
            type3_x.append(datingDataMat[index][0])
            type3_y.append(datingDataMat[index][1])
        index += 1

    type1 = ax.scatter(type1_x, type1_y, s=30, c='b')
    type2 = ax.scatter(type2_x, type2_y, s=40, c='r')
    type3 = ax.scatter(type3_x, type3_y, s=50, c='y', marker=(3,1))

    '''
     scatter是用来画散点图的
     matplotlib.pyplot.scatter(x, y, s=20, c='b', marker='o', cmap=None, norm=None, vmin=None, vmax=None, alpha=None, linewidths=None, verts=None, hold=None,**kwargs)
     其中，xy是点的坐标，s点的大小
     maker是形状可以maker=（5，1）5表示形状是5边型，1表示是星型（0表示多边形，2放射型，3圆形）
     alpha表示透明度；facecolor=‘none’表示不填充。
    '''

    ax.legend((type1, type2, type3), ('不喜欢', '魅力一般', '极具魅力'), loc=0)
    '''
    loc(设置图例显示的位置)
    'best'         : 0, (only implemented for axes legends)(自适应方式)
    'upper right'  : 1,
    'upper left'   : 2,
    'lower left'   : 3,
    'lower right'  : 4,
    'right'        : 5,
    'center left'  : 6,
    'center right' : 7,
    'lower center' : 8,
    'upper center' : 9,
    'center'       : 10,
    '''
    plt.show()
