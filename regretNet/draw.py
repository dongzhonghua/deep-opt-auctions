# -*- coding: utf-8 -*-#
#-------------------------------------------------------------------------------
# Name:         draw
# Date:         2019/6/17
#-------------------------------------------------------------------------------
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import json

from scipy.interpolate import BSpline

from NonLinearRegression import regression

plt.rcParams['font.sans-serif']=['Arial']   #如果要显示中文字体，则在此处设为：SimHei
plt.rcParams['axes.unicode_minus']=False    #显示负号
with open("5x5data.json") as f1:
    data1 = f1.read()
    json1 = json.loads(data1)
    rev1 = [x*8/3 for x in json1["Revenue"]]
    reg1 = json1["Regret"]
    regl1 = json1["Reg_Loss"]
    netl1 = json1["Net_Loss"]
with open("10x1data.json") as f2:
    data2 = f2.read()
    json2 = json.loads(data2)
    rev2 = [x*8/3 for x in json2["Revenue"]]
    reg2 = json2["Regret"]
    regl2 = json2["Reg_Loss"]
    netl2 = json2["Net_Loss"]
with open("5x1data.json") as f3:
    data3 = f3.read()
    json3 = json.loads(data3)
    rev3 = [x*8/3 for x in json3["Revenue"]]
    reg3 = json3["Regret"]
    regl3 = json3["Reg_Loss"]
    netl3 = json3["Net_Loss"]
with open("10x5data.json") as f4:
    data4 = f4.read()
    json4 = json.loads(data4)
    rev4 = [x*8/3 for x in json4["Revenue"]]
    reg4 = json4["Regret"]
    regl4 = json4["Reg_Loss"]
    netl4 = json4["Net_Loss"]
with open("10x10data.json") as f5:
    data5 = f5.read()
    json5 = json.loads(data5)
    rev5 = [x*8/3 for x in json5["Revenue"]]
    reg5 = json5["Regret"]
    regl5 = json5["Reg_Loss"]
    netl5 = json5["Net_Loss"]
with open("15x5data.json") as f6:
    data6 = f6.read()
    json6 = json.loads(data6)
    rev6 = [x*8/3 for x in json6["Revenue"]]
    reg6 = json6["Regret"]
    regl6 = json6["Reg_Loss"]
    netl6 = json6["Net_Loss"]
with open("10x15data.json") as f7:
    data7 = f7.read()
    json7 = json.loads(data7)
    rev7 = [x*8/3 for x in json7["Revenue"]]
    reg7 = json7["Regret"]
    regl7 = json7["Reg_Loss"]
    netl7 = json7["Net_Loss"]
with open("15x1data.json") as f8:
    data9 = f8.read()
    json9 = json.loads(data9)
    rev9 = [x*8/3 for x in json9["Revenue"]]
    reg9 = json9["Regret"]
    regl9 = json9["Reg_Loss"]
    netl9 = json9["Net_Loss"]
x = np.array([x*10  for x in range(600)])

def show_revenue1():

    # label在图示(legend)中显示。若为数学公式，则最好在字符串前后添加"$"符号
    # color：b:blue、g:green、r:red、c:cyan、m:magenta、y:green、k:black、w:white、、、
    # 线型：-    --     -.    :       ,
    # marker：.    ,      o      v        <       *        +        1
    # plt.figure(figsize=(7,7))
    plt.grid(linestyle="--")  # 设置背景网格线为虚线
    ax = plt.gca()
    # ax.spines['top'].set_visible(False)   #去掉上边框
    # ax.spines['right'].set_visible(False) #去掉右边框
    # regression(x,[rev1[i*10] for i in range(600)])

    plt.plot(x, [rev1[i*10] for i in range(600)], color="blue", label="5 bidders and 5 items", linewidth=1.0)
    plt.plot(x, [5.86 for _ in range(600)], color="blue", linestyle="--", label="BaseLine of 5 bidders and 5 items", linewidth=2.0)
    plt.plot(x, [rev4[i*10] for i in range(600)], "green", label="10 bidders and 5 items", linewidth=1.0)
    plt.plot(x, [6.11 for _ in range(600)], "green",linestyle="--", label="BaseLine of 10 bidders and 5 items", linewidth=2.0)
    plt.plot(x, [rev6[i*10] for i in range(600)], color="red", label="15 bidders and 5 items", linewidth=1.0)
    plt.plot(x, [6.24 for _ in range(600)], color="red",linestyle="--", label="BaseLine of 15 bidders and 5 items", linewidth=2.0)

    # group_labels=['dataset1','dataset2','dataset3','dataset4','dataset5',' dataset6','dataset7','dataset8','dataset9','dataset10'] #x轴刻度的标识
    # plt.xticks(x,group_labels,fontsize=12,fontweight='bold')  #默认字体大小为10
    plt.yticks(fontsize=12)
    # plt.title("Test",fontsize=12,fontweight='bold')        #默认字体大小为12
    plt.xlabel("No. of iterations", fontsize=13)
    plt.ylabel("Revenue", fontsize=13)
    plt.xlim(0, 6000)  # 设置x轴的范围
    # plt.ylim(0,9)
    # plt.legend()                   #显示各曲线的图例
    plt.legend(loc=0, numpoints=1)
    leg = plt.gca().get_legend()
    ltext = leg.get_texts()
    plt.setp(ltext, fontsize=10)  # 设置图例字体的大小和粗细
    plt.savefig('revenue1.pdf', format='pdf')  # 建议保存为pdf格式，再用inkscape转为矢量图emf后插入word中
    plt.show()

def show_revenue2():

    # label在图示(legend)中显示。若为数学公式，则最好在字符串前后添加"$"符号
    # color：b:blue、g:green、r:red、c:cyan、m:magenta、y:yellow、k:black、w:white、、、
    # 线型：-    --     -.    :       ,
    # marker：.    ,      o      v        <       *        +        1
    # plt.figure(figsize=(7,7))
    plt.grid(linestyle="--")  # 设置背景网格线为虚线
    ax = plt.gca()
    # ax.spines['top'].set_visible(False)   #去掉上边框
    # ax.spines['right'].set_visible(False) #去掉右边框
    plt.plot(x, [rev4[i*10] for i in range(600)], color="blue", label="10 bidders and 5 items", linewidth=1.0)
    plt.plot(x, [6.11 for _ in range(600)], color="blue", linestyle="--", label="BaseLine of 10 bidders and 5 items", linewidth=2.0)
    plt.plot(x, [rev5[i*10] for i in range(600)], "green", label="10 bidders and 10 items", linewidth=1.0)
    plt.plot(x, [12.81 for _ in range(600)], "green",linestyle="--", label="BaseLine of 10 bidders and 10 items", linewidth=2.0)
    plt.plot(x, [rev7[i*10] for i in range(600)], color="red", label="10 bidders and 15 items", linewidth=1.0)
    plt.plot(x, [18.56 for _ in range(600)], color="red",linestyle="--", label="BaseLine of 10 bidders and 15 items", linewidth=2.0)

    # group_labels=['dataset1','dataset2','dataset3','dataset4','dataset5',' dataset6','dataset7','dataset8','dataset9','dataset10'] #x轴刻度的标识
    # plt.xticks(x,group_labels,fontsize=12,fontweight='bold')  #默认字体大小为10
    plt.yticks(fontsize=12)
    # plt.title("Test",fontsize=12,fontweight='bold')        #默认字体大小为12
    plt.xlabel("No. of iterations", fontsize=13)
    plt.ylabel("Revenue", fontsize=13)
    plt.xlim(0, 6000)  # 设置x轴的范围
    # plt.ylim(0,9)
    # plt.legend()                   #显示各曲线的图例
    plt.legend(loc=0, numpoints=1)
    leg = plt.gca().get_legend()
    ltext = leg.get_texts()
    plt.setp(ltext, fontsize=10)  # 设置图例字体的大小和粗细
    plt.savefig('revenue2.pdf', format='pdf')  # 建议保存为pdf格式，再用inkscape转为矢量图emf后插入word中
    plt.show()



def show_revenue3():

    # label在图示(legend)中显示。若为数学公式，则最好在字符串前后添加"$"符号
    # color：b:blue、g:green、r:red、c:cyan、m:magenta、y:yellow、k:black、w:white、、、
    # 线型：-    --     -.    :       ,
    # marker：.    ,      o      v        <       *        +        1
    # plt.figure(figsize=(7,7))
    plt.grid(linestyle="--")  # 设置背景网格线为虚线
    ax = plt.gca()
    # ax.spines['top'].set_visible(False)   #去掉上边框
    # ax.spines['right'].set_visible(False) #去掉右边框
    # plt.plot(x, rev3, color="blue", label="5 bidders and 1 items")
    # plt.plot(x, [1.86 for _ in range(6000)], color="blue", linestyle="--", label="BaseLine of 5 bidders and 1 items", linewidth=1.0)
    plt.plot(x, [rev2[i*10] for i in range(600)], "green", label="10 bidders and 1 item",linewidth=1.0)
    plt.plot(x, [1.65 for _ in range(600)], "green",linestyle="--", label="Revenue of sigle-item auction", linewidth=2.0)
    plt.plot(x, [rev9[i*10] for i in range(600)], "red", label="15 bidders and 1 item",linewidth=1.0)
    plt.plot(x, [1.71 for _ in range(600)], "red",linestyle="--", label="Revenue of sigle-item auction", linewidth=2.0)

    # group_labels=['dataset1','dataset2','dataset3','dataset4','dataset5',' dataset6','dataset7','dataset8','dataset9','dataset10'] #x轴刻度的标识
    # plt.xticks(x,group_labels,fontsize=12,fontweight='bold')  #默认字体大小为10
    plt.yticks(fontsize=12)
    # plt.title("Test",fontsize=12,fontweight='bold')        #默认字体大小为12
    plt.xlabel("No. of iterations", fontsize=13)
    plt.ylabel("Revenue", fontsize=13)
    plt.xlim(0, 6000)  # 设置x轴的范围
    # plt.ylim(0,9)
    # plt.legend()                   #显示各曲线的图例
    plt.legend(loc=4, numpoints=1)
    leg = plt.gca().get_legend()
    ltext = leg.get_texts()
    plt.setp(ltext, fontsize=10)  # 设置图例字体的大小和粗细
    plt.savefig('revenue3.pdf', format='pdf')  # 建议保存为pdf格式，再用inkscape转为矢量图emf后插入word中
    plt.show()


def show_regret():
    plt.grid(linestyle="--")  # 设置背景网格线为虚线
    ax = plt.gca()
    # ax.spines['top'].set_visible(False)   #去掉上边框
    # ax.spines['right'].set_visible(False) #去掉右边框
    # fig,ax = plt.subplots(2,2)
    # fig.suptitle("Regret")
    # plt.figure(figsize=(10,7))
    plt.ylim(0, 0.4)
    plt.xlim(0, 6000)
    plt.xlabel("No. of iterations", fontsize=13)
    plt.ylabel("Regret probability", fontsize=13)
    plt.plot(x, [reg1[i*10]*10 for i in range(600)], color="black", label="5 bidders and 5 items", linewidth=1.0)
    plt.plot(x, [reg4[i*10]*10 for i in range(600)], color="green", label="10 bidders and 5 items", linewidth=1.0)

    plt.plot(x, [reg5[i*10]*10 for i in range(600)], color="red", label="10 bidders and 10 items", linewidth=1.0)
    plt.plot(x, [reg7[i*10]*10 for i in range(600)], color="blue", label="10 bidders and 15 items", linewidth=1.0)
    # group_labels=['dataset1','dataset2','dataset3','dataset4','dataset5',' dataset6','dataset7','dataset8','dataset9','dataset10'] #x轴刻度的标识
    # plt.xticks(x,group_labels,fontsize=12,fontweight='bold')  #默认字体大小为10
    plt.yticks(fontsize=12)
    # plt.title("Test",fontsize=12,fontweight='bold')        #默认字体大小为12
    plt.legend(loc=0, numpoints=1)
    leg = plt.gca().get_legend()
    ltext = leg.get_texts()
    plt.setp(ltext, fontsize=10)  # 设置图例字体的大小和粗细
    # plt.xlim(0, 10000)  # 设置x轴的范围
    # # plt.ylim(0,9)
    # # plt.legend()                   #显示各曲线的图例
    # plt.legend(loc=0, numpoints=1)
    # leg = plt.gca().get_legend()
    # ltext = leg.get_texts()
    # plt.setp(ltext, fontsize=10)  # 设置图例字体的大小和粗细
    plt.savefig('regret.pdf', format='pdf')  # 建议保存为pdf格式，再用inkscape转为矢量图emf后插入word中
    plt.show()
show_revenue1()
show_revenue2()
show_revenue3()
show_regret()
