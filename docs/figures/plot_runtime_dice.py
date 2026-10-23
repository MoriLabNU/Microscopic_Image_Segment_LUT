import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 方法名和相应的数据
methods = ['Otsu', 'Triangle', 'SR-CNN', 'SEG-CNN', 'U-Net', 'SEG-LUT']
runtime = [15.31, 16.12, 2135.04, 3207.94, 16945.75, 736.66]  # 运行时间，单位为毫秒
dice = [54.28, 62.89, 77.72, 82.60, 88.05, 80.43]  # Dice系数

# 分别为不同的方法组设置标记和颜色
markers = ['o', 'o', '^', '^', '^', '*']  # o: 圆圈, ^: 三角形, *: 星星
colors = ['#B07100', '#B07100', '#486F93', '#486F93', '#486F93', '#843C0C']  # 不同的颜色表示不同的组
sizes = [100, 100, 100, 100, 100, 200]  # 调整最后一个值以增大星星标记的大小
# 创建图表
plt.figure(figsize=(10, 5), facecolor='#FFFFF0')
#plt.figure(figsize=(10, 5))
# 获取当前轴（ax）并设置网格线在数据点下方
ax = plt.gca()
ax.set_axisbelow(True)

ax.set_facecolor('#FFFFF0')  # 将绘图区域的背景色设置为淡蓝色

# 绘制每个方法的点并添加文本标签，同时微调文本位置以避免重叠
text_offset_x = 300  # X轴的文本偏移量
text_offset_y = 0    # Y轴的文本偏移量

# for i, method in enumerate(methods):
#     plt.scatter(runtime[i], dice[i], marker=markers[i], color=colors[i], s=sizes[i])
#     plt.text(runtime[i] + text_offset_x, dice[i] + text_offset_y, f'{method}', verticalalignment='bottom', horizontalalignment='left', fontsize=9)

for i, method in enumerate(methods):
    plt.scatter(runtime[i], dice[i], marker=markers[i], color=colors[i], s=sizes[i])
    if method == "SEG-LUT":
        # 对"SEG-LUT"加粗并保持黑色
        plt.text(runtime[i] + text_offset_x, dice[i] + text_offset_y, f'{method}',
                 verticalalignment='center', horizontalalignment='left',
                 fontsize=17, fontweight='bold',color='#843C0C')
    elif method == "SEG-CNN":
        # 对"SEG-CNN"加粗并保持黑色
        plt.text(runtime[i] + text_offset_x, dice[i] + text_offset_y, f'{method}',
                 verticalalignment='center', horizontalalignment='left',
                 fontsize=15, fontweight='bold',color='black')
    else:
        # 其他方法使用深灰色并保持默认权重
        plt.text(runtime[i] + text_offset_x, dice[i] + text_offset_y, f'{method}',
                 verticalalignment='center', horizontalalignment='left',
                 fontsize=15, color='black')

# 添加标题和坐标轴标签
plt.title('Runtime and Dice Score for Different Methods', fontsize=16)  # 标题字体大小
plt.xlabel('Runtime (ms)', fontsize=16)  # X轴标签字体大小
plt.ylabel('Dice Score (%)', fontsize=16)  # Y轴标签字体大小
plt.xlim(left=plt.xlim()[0], right=20000)
# 显示网格（可选）
plt.grid(True)
plt.xticks(fontsize=12)  # X轴刻度标签字体大小
plt.yticks(fontsize=12)  # Y轴刻度标签字体大小
# 显示图例
#plt.legend(loc='lower right')

# 保存图表到文件，而不是显示它
# 你可以通过更改 'runtime_vs_dice.png' 来指定不同的文件名和路径
# dpi参数控制图像的分辨率
_out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runtime_vs_dice2.png')
plt.savefig(_out, dpi=200)


