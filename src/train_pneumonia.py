import tensorflow as tf
from tensorflow.keras import layers, models, applications
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import class_weight
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

# ========== 修改这里！ ==========
data_dir = "C:/Users/86191/data/chest_xray/chest_xray"  # 正确路径
img_height = 224  # 增加图像尺寸以提高特征提取能力
img_width = 224   
batch_size = 16   # 批次大小（GPU内存小则调小）

# 加载训练集（自动划分20%为验证集）
raw_train_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(data_dir, 'train'),
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

# 加载验证集
raw_val_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(data_dir, 'train'),
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

# 加载测试集（独立测试集）
test_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(data_dir, 'test'),
    image_size=(img_height, img_width),
    batch_size=batch_size
)

# 获取类别名称（正常/NORMAL、肺炎/PNEUMONIA）
class_names = raw_train_ds.class_names
print("数据集类别：", class_names)

# 统计训练集样本分布
normal_count = sum(1 for _, label in raw_train_ds.unbatch() if label.numpy() == 0)
pneumonia_count = sum(1 for _, label in raw_train_ds.unbatch() if label.numpy() == 1)
total_samples = normal_count + pneumonia_count
print(f"训练集样本分布 - 正常: {normal_count}, 肺炎: {pneumonia_count}")
print(f"类别比例: 正常: {normal_count/total_samples:.2%}, 肺炎: {pneumonia_count/total_samples:.2%}")

# 数据增强层 - 针对医学图像的优化增强（减小增强强度）
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.01),  # 非常小的旋转
    layers.RandomZoom(0.01),      # 非常小的缩放
    layers.RandomContrast(0.05),
])

# 应用数据增强到训练集（先缓存原始数据再增强）
AUTOTUNE = tf.data.AUTOTUNE

# 缓存原始数据集
raw_train_ds = raw_train_ds.cache()
raw_val_ds = raw_val_ds.cache()
test_ds = test_ds.cache()

# 对训练集应用动态增强
train_ds = raw_train_ds.map(
    lambda x, y: (data_augmentation(x, training=True), y),
    num_parallel_calls=AUTOTUNE
)

# 优化数据加载性能
train_ds = train_ds.shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = raw_val_ds.prefetch(buffer_size=AUTOTUNE)

# 可视化前9张图像（增强后的数据）
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
plt.show()

# 使用更轻量的预训练模型（DenseNet121更适合医学图像）
base_model = applications.DenseNet121(
    weights='imagenet',  # 使用ImageNet预训练权重
    include_top=False,   # 不包含顶部的全连接层
    input_shape=(img_height, img_width, 3)
)

# 冻结基础模型层
base_model.trainable = False

# 构建更简单的模型 - 添加标准化层！
model = models.Sequential([
    # 添加标准化层（关键修复！将[0,255] -> [-1,1]）
    layers.Rescaling(1./127.5, offset=-1, input_shape=(img_height, img_width, 3)),
    
    # 预训练模型作为特征提取器
    base_model,
    
    # 全局平均池化
    layers.GlobalAveragePooling2D(),
    
    # 批归一化
    layers.BatchNormalization(),
    
    # 全连接层 - 简化结构
    layers.Dropout(0.5),  # 增加Dropout防止过拟合
    
    # 更小的全连接层
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    
    # 输出层
    layers.Dense(1, activation='sigmoid')
])

# 使用更保守的学习率
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),  # 更低的学习率
    loss='binary_crossentropy',
    metrics=[
        'accuracy',
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='auc')  # AUC指标
    ]
)

# 计算更加强调的类别权重（解决数据不平衡问题）
class_weights = {
    0: (1/normal_count) * (total_samples)/2.0,  # 增加正常样本的权重
    1: (1/pneumonia_count) * (total_samples)/2.0
}
print(f"调整后的类别权重: {class_weights}")

# 学习率调度
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=3,
    min_lr=1e-6,
    verbose=1
)

# 早停回调
early_stopping = EarlyStopping(
    monitor='val_auc',  # 监控AUC指标
    patience=10,  # 增加耐心值
    restore_best_weights=True,
    mode='max'  # AUC越大越好
)

# 添加模型检查点回调
checkpoint = ModelCheckpoint(
    'best_model.h5',
    save_best_only=True,
    monitor='val_auc',
    mode='max',
    verbose=1
)

# 训练模型（只训练顶层）
epochs = 20  # 减少epoch数量
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    callbacks=[reduce_lr, early_stopping, checkpoint],
    class_weight=class_weights,
    verbose=1
)

# 加载最佳模型
model.load_weights('best_model.h5')
print("加载验证集表现最佳模型完成")

# 解冻基础模型的部分层进行微调
base_model.trainable = True
# 冻结大部分层，只训练最后几层
for layer in base_model.layers[:-20]:
    layer.trainable = False

# 重新编译模型，使用更小的学习率
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-6),  # 微小的学习率
    loss='binary_crossentropy',
    metrics=[
        'accuracy',
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='auc')
    ]
)

# 微调模型 - 创建新的回调实例（避免状态继承）
print("\n开始微调模型...")
fine_tune_epochs = 10
total_epochs = epochs + fine_tune_epochs

# 新的回调实例
reduce_lr_fine = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=3,
    min_lr=1e-7,  # 更小的学习率
    verbose=1
)

early_stopping_fine = EarlyStopping(
    monitor='val_auc',
    patience=8,
    restore_best_weights=True,
    mode='max'
)

history_fine = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=total_epochs,
    initial_epoch=history.epoch[-1],  # 从上一次训练结束处开始
    callbacks=[reduce_lr_fine, early_stopping_fine, checkpoint],
    class_weight=class_weights,
    verbose=1
)

# 再次加载最佳模型（微调后的）
model.load_weights('best_model.h5')
print("加载微调后最佳模型完成")

# 测试集评估
test_loss, test_acc, test_precision, test_recall, test_auc = model.evaluate(test_ds, verbose=2)
print(f"\n测试集评估结果:")
print(f"准确率: {test_acc * 100:.2f}%")
print(f"精确率: {test_precision * 100:.2f}%")
print(f"召回率: {test_recall * 100:.2f}%")
print(f"AUC: {test_auc * 100:.2f}%")

# 计算F1分数
f1_score = 2 * (test_precision * test_recall) / (test_precision + test_recall + 1e-7)
print(f"F1分数: {f1_score * 100:.2f}%")

# 合并训练历史
acc = history.history['accuracy'] + history_fine.history['accuracy']
val_acc = history.history['val_accuracy'] + history_fine.history['val_accuracy']

loss = history.history['loss'] + history_fine.history['loss']
val_loss = history.history['val_loss'] + history_fine.history['val_loss']

precision = history.history['precision'] + history_fine.history['precision']
val_precision = history.history['val_precision'] + history_fine.history['val_precision']

recall = history.history['recall'] + history_fine.history['recall']
val_recall = history.history['val_recall'] + history_fine.history['val_recall']

auc = history.history['auc'] + history_fine.history['auc']
val_auc = history.history['val_auc'] + history_fine.history['val_auc']

# 绘制训练曲线
plt.figure(figsize=(16, 12))

plt.subplot(2, 2, 1)
plt.plot(acc, label='训练准确率')
plt.plot(val_acc, label='验证准确率')
plt.axvline(x=epochs-1, color='r', linestyle='--', label='开始微调')
plt.legend()
plt.title('准确率变化')

plt.subplot(2, 2, 2)
plt.plot(loss, label='训练损失')
plt.plot(val_loss, label='验证损失')
plt.axvline(x=epochs-1, color='r', linestyle='--', label='开始微调')
plt.legend()
plt.title('损失变化')

plt.subplot(2, 2, 3)
plt.plot(precision, label='训练精确率')
plt.plot(val_precision, label='验证精确率')
plt.axvline(x=epochs-1, color='r', linestyle='--', label='开始微调')
plt.legend()
plt.title('精确率变化')

plt.subplot(2, 2, 4)
plt.plot(recall, label='训练召回率')
plt.plot(val_recall, label='验证召回率')
plt.axvline(x=epochs-1, color='r', linestyle='--', label='开始微调')
plt.legend()
plt.title('召回率变化')

plt.tight_layout()
plt.show()

# 保存优化后的模型
model.save('optimized_pneumonia_model.h5')

# 测试单张正常样本和肺炎样本
def test_sample(image_path, label):
    # 检查文件是否存在
    if not os.path.exists(image_path):
        print(f"\n错误：文件不存在 - {image_path}")
        return
    
    try:
        img = tf.keras.utils.load_img(image_path, target_size=(img_height, img_width))
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)  # 不要除以255！（关键修复）
        
        prediction = model.predict(img_array)
        pred_value = prediction[0][0]
        pred_class = "肺炎" if pred_value > 0.5 else "正常"
        confidence = pred_value if pred_class == "肺炎" else (1 - pred_value)
        
        print(f"\n测试样本: {os.path.basename(image_path)} (实际: {label})")
        print(f"预测类别: {pred_class}")
        print(f"预测值: {pred_value:.6f}")
        print(f"置信度: {confidence:.2%}")
        print(f"{'✓ 正确' if pred_class == label else '✗ 错误'}")
        
        # 返回预测值用于后续分析
        return pred_value
    except Exception as e:
        print(f"\n测试失败: {str(e)}")
        return None

# 测试一张正常样本
print("\n测试正常样本:")
normal_sample = os.path.join(data_dir, 'test/NORMAL/IM-0001-0001.jpeg')
normal_pred = test_sample(normal_sample, "正常")

# 测试一张肺炎样本
print("\n测试肺炎样本:")
pneumonia_sample = os.path.join(data_dir, 'test/PNEUMONIA/person1_virus_6.jpeg')
pneumonia_pred = test_sample(pneumonia_sample, "肺炎")

# 额外诊断：测试多个样本以确认模型行为
print("\n额外诊断：测试多个样本")
normal_dir = os.path.join(data_dir, 'test/NORMAL')
pneumonia_dir = os.path.join(data_dir, 'test/PNEUMONIA')

# 随机选择5个正常样本
normal_files = [os.path.join(normal_dir, f) for f in os.listdir(normal_dir) if f.endswith('.jpeg')][:5]
# 随机选择5个肺炎样本
pneumonia_files = [os.path.join(pneumonia_dir, f) for f in os.listdir(pneumonia_dir) if f.endswith('.jpeg')][:5]

print("\n测试5个正常样本:")
for file in normal_files:
    pred = test_sample(file, "正常")
    if pred is not None and pred > 0.5:
        print(f"警告: 正常样本被错误分类为肺炎: {os.path.basename(file)}")

print("\n测试5个肺炎样本:")
for file in pneumonia_files:
    pred = test_sample(file, "肺炎")
    if pred is not None and pred <= 0.5:
        print(f"警告: 肺炎样本被错误分类为正常: {os.path.basename(file)}")