import tensorflow as tf
import matplotlib.pyplot as plt
from time import *


# 数据集加载函数，指明数据集的位置并统一处理为imgheight*imgwidth的大小，同时设置batch
def data_load(data_dir, test_data_dir, img_height, img_width, batch_size):
    # 加载训练集
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        label_mode='categorical',
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)
    # 加载测试集
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        test_data_dir,
        label_mode='categorical',
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)
    class_names = train_ds.class_names
    # 返回处理之后的训练集、验证集和类名
    return train_ds, val_ds, class_names


# 构建CNN模型
def model_load(IMG_SHAPE=(224, 224, 3), class_num=2):
    # 搭建模型
    model = tf.keras.models.Sequential([
    tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=IMG_SHAPE),  # 归一化，将像素值处理成0到1之间的值
    tf.keras.layers.Conv2D(filters=48, kernel_size=11, strides=4, padding="same", activation="relu"),
    tf.keras.layers.BatchNormalization(name='BN1'),
    tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
    tf.keras.layers.Conv2D(filters=128, kernel_size=5, padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(name='BN2'),
    tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
    tf.keras.layers.Conv2D(filters=192, kernel_size=3, padding='same', activation='relu'),
    tf.keras.layers.Conv2D(filters=86, kernel_size=3, padding='same', activation='relu'),
    tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(name='BN3'),
    tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(class_num, activation='sigmoid')
])

    # 输出模型信息
    model.summary()
    # 指明模型的训练参数，优化器为sgd优化器，损失函数为交叉熵损失函数
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # 返回模型
    return model


# 展示训练过程的曲线
def show_loss_acc(history):
    # 从history中提取模型训练集和验证集准确率信息和误差信息
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    # 按照上下结构将图画输出
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()), 1])
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.savefig('results/results_AlexNet_test5.png', dpi=100)


def train(epochs):
    # 开始训练，记录开始时间
    begin_time = time()
    # todo 加载数据集， 修改为你的数据集的路径
    train_ds, val_ds, class_names = data_load(r"D:\Python\PythonProject\Truck_identification\Final_data\train",
                                              r"D:\Python\PythonProject\Truck_identification\Final_data\val", 224, 224, 16)
    print(class_names)
    # 加载模型
    model = model_load(class_num=len(class_names))
    # 指明训练的轮数epoch，开始训练
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)
    # todo 保存模型， 修改为你要保存的模型的名称
    model.save("models/AlexNet_test5.h5")
    # 记录结束时间
    end_time = time()
    run_time = end_time - begin_time
    print('该循环程序运行时间：', run_time, "s")
    # 绘制模型训练过程图
    show_loss_acc(history)


if __name__ == '__main__':
    train(epochs=100)
