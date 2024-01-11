import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import TensorBoard
import datetime

# 生成随机训练集
np.random.seed(42)
x_train = np.random.rand(100, 7)
y_train = np.random.rand(100, 3)

# 创建神经网络模型
model = Sequential()
model.add(Dense(10, input_dim=7, activation='relu'))
model.add(Dense(3, activation='linear'))  # 多目标回归问题

# 编译模型
model.compile(optimizer='adam', loss='mean_squared_error')

# 创建TensorBoard回调
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True, write_images=True)

# 训练模型，并将TensorBoard回调传递给fit方法
model.fit(x_train, y_train, epochs=10, callbacks=[tensorboard_callback])

# 保存模型
model.save('my_model.keras')





