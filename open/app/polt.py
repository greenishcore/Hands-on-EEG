import numpy as np
import vispy.scene
from vispy.scene import visuals
import pandas as pd

# 读取数据
df = pd.read_csv("C:\\Users\\a1882\\Desktop\\EEG\\new_implement\\data\\origin_raw_data\\lefthand_zyy_04_epocflex_2023.03.22t16.48.29+08.00.md.bp.csv",header=None)
data_c = df.iloc[1:,1:].astype(float).values.T

# 创建可视化场景
canvas = vispy.scene.SceneCanvas(keys='interactive', show=True)
view = canvas.central_widget.add_view()

# 创建32个线条对象
lines = []
for i in range(32):
    line = visuals.Line(pos=np.array([[0, 0], [1, 1]]), width=2, color='#00251C')
    lines.append(line)
    view.add(line)

# 设置坐标轴范围和标签
view.camera.rect = (0, -800, data_c.shape[1], 800)
view.camera.scale = (1, 0.1)
view.camera.flip = (False, True)
view.camera.set_range()

view.camera.axis = 'left'
view.camera.axis_label = 'Amplitude (uV)'
view.camera.axis_direction = (1, -1)

view.camera._axis._font_size = 12
view.camera._axis._tick_label_margin = 20

view.camera._axis._major_tick_in = 5
view.camera._axis._major_tick_out = 10
view.camera._axis._minor_tick_in = 2
view.camera._axis._minor_tick_out = 5

view.camera._axis._label_margin = 40
view.camera._axis._tick_label_margin = 20
view.camera._axis._visible = True

# 添加标题
title = vispy.scene.visuals.Text('32-Channel EEG Data', font_size=16, anchor_x='center', anchor_y='top', pos=[view.camera.rect[2] / 2, view.camera.rect[1] - 50])
view.add(title)

# 创建滑窗高亮动画
window_size = 128
start_pos = 0
end_pos = start_pos + window_size - 1

def update(event):
    global start_pos, end_pos
    lines_pos = []
    for i in range(32):
        line_pos = data_c[i, start_pos:end_pos+1]
        line_pos = np.stack([np.arange(window_size), line_pos], axis=1)
        line_pos[:, 0] += start_pos
        lines_pos.append(line_pos)
    for i in range(32):
        lines[i].set_data(pos=lines_pos[i])
    start_pos += 1
    end_pos += 1
    if end_pos >= data_c.shape[1]:
        start_pos = 0
        end_pos = start_pos + window_size - 1

timer = vispy.app.Timer(connect=update, interval=0.01)
timer.start()

if __name__ == '__main__':
    vispy.app.run()
