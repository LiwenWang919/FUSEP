

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator
import numpy as np

plt.rcParams['font.family'] = 'Times New Roman'

# epoch,acc,loss,val_acc,val_loss
x_axis_data = [1, 5, 10, 20, 30, 40]
y_axis_data1 = [56.1, 58.5, 60.3, 60.6, 60.4, 59.4]  # 10%
y_axis_data2 = [50.5, 53.2, 54, 55.9, 55.2, 54.7]  # 5%
y_axis_data3 = [26.3, 28.2, 28.8, 29.2, 29.4, 26.4]  # 1%


# 画图
plt.plot(x_axis_data, y_axis_data1, marker='o', alpha=1, linewidth=1.5, label='labelled 10%', color='green')
plt.plot(x_axis_data, y_axis_data2, marker='s', alpha=1, linewidth=1.5, label='labelled 5%', color='orange')
plt.plot(x_axis_data, y_axis_data3, marker='^', alpha=1, linewidth=1.5, label='labelled 1%', color='blue')

plt.xlabel('Effect of reliable ratio ' + r'$C$', fontsize=12)
plt.ylabel('mAP(%)', fontsize=12)  # accuracy
plt.grid(alpha=0.3, linestyle='--')

# plt.yaxis.set_major_locator(FixedLocator([10, 20, 50]))


# 设置数据标签位置及大小
for a, b in zip(x_axis_data, y_axis_data1):
    plt.text(a, b, str(b), ha='center', va='bottom', fontsize=12)  # ha='center', va='top'
for a, b1 in zip(x_axis_data, y_axis_data2):
    plt.text(a, b1, str(b1), ha='center', va='bottom', fontsize=12)
for a, b2 in zip(x_axis_data, y_axis_data3):
    plt.text(a, b2, str(b2), ha='center', va='bottom', fontsize=12)


plt.legend()  # 显示上面的label
plt.savefig('reliable_rato.png',dpi=300)



import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'Times New Roman'
 
#epoch,acc,loss,val_acc,val_loss
x_axis_data = [5,10,20,50,100]
y_axis_data1 = [54.5,58.1,60.6,61.4,62.2] # 10%
y_axis_data2 = [51.4,53.2,55.9,56.3,56.9] # 5%
y_axis_data3 = [26.2,27.8,29.2,29.6,30.1] # 1%

        
#画图 
plt.plot(x_axis_data, y_axis_data1, marker='o', alpha=1, linewidth=1.5, label='labelled 10%', color='green')#'
plt.plot(x_axis_data, y_axis_data2, marker='s', alpha=1, linewidth=1.5, label='labelled 5%',  color='orange')
plt.plot(x_axis_data, y_axis_data3, marker='^', alpha=1, linewidth=1.5, label='labelled 1%', color='blue')


plt.xlabel('Effect of queue length ' + r'$K$', fontsize=12)
plt.ylabel('mAP(%)', fontsize=12)#accuracy
plt.grid(alpha=0.3, linestyle='--')

## 设置数据标签位置及大小
for a, b in zip(x_axis_data, y_axis_data1):
    plt.text(a, b, str(b), ha='center', va='bottom', fontsize=12)  #  ha='center', va='top'
for a, b1 in zip(x_axis_data, y_axis_data2):
    plt.text(a, b1, str(b1), ha='center', va='bottom', fontsize=12)  
for a, b2 in zip(x_axis_data, y_axis_data3):
    plt.text(a, b2, str(b2), ha='center', va='bottom', fontsize=12)
plt.legend()  #显示上面的label



 
#plt.ylim(-1,1)#仅设置y轴坐标范围
plt.savefig('queue_length.png', dpi=300)