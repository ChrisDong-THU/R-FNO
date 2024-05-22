import pandas as pd
import matplotlib.pyplot as plt

# 读取CSV文件
train_loss_df = pd.read_csv('train_loss.csv')
val_loss_df = pd.read_csv('val_loss.csv')

# 确保epoch作为X轴
epoch_train = train_loss_df.iloc[:, 0]
loss_train = train_loss_df.iloc[:, 1]

epoch_val = val_loss_df.iloc[:, 0]
loss_val = val_loss_df.iloc[:, 1]

# 绘制损失曲线
plt.figure(figsize=(10, 5))
plt.plot(epoch_train, loss_train, label='Train Loss')
plt.plot(epoch_val, loss_val, label='Validation Loss')

# 设置y轴为对数缩放
plt.yscale('log')

# 添加图例
plt.legend()

# 添加图表标题和轴标签
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss (log scale)')

# 显示图表
plt.show()