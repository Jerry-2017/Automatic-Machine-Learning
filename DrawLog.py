import matplotlib.pyplot as plt
import sys
import numpy as np

def clip_after(passage,word):
    pos=passage.find(word)
    pos+=len(word)
    return passage[pos:]

def draw_log_data(path):
    f=open(path,'r')
    acc_list=[]
    step_list=[]
    cnt=0
    for i in f.readlines():
        if i.find("Task Net Acc")>0:
            print(i)
            acc=float(clip_after(i,"Task Net Acc "))
            acc_list.append(acc)
            cnt+=1
            step_list.append(cnt)
    return acc_list,step_list

path=sys.argv[1]
acc,step=draw_log_data(path)
fig = plt.figure()
ax = fig.add_subplot(111)
t1 = np.arange(0, len(step), 1)

ax.scatter(step, acc,s=1)



fig.savefig("1.png")