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
    resNet_base = tf.keras.applications.ResNet50(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')
    resNet_base.trainable = True
    # for layer in resNet_base.layers[:-100]:
    #     layer.trainable = False
    model = tf.keras.models.Sequential([
    resNet_base,
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2(0.1)),
    tf.keras.layers.Dropout(0.6),
    tf.keras.layers.Dense(class_num, activation='softmax'),
    ])
    # 输出模型信息
    model.summary()
    # 指明模型的训练参数，优化器为sgd优化器，损失函数为交叉熵损失函数
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
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
    plt.savefig('results/results_ResNet.png', dpi=100)

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy')==1 and logs.get('val_accuracy')==1):
            print("\n模型已达到最优！")
            self.model.stop_training = True

callbacks = myCallback()

def train(epochs):
    # 开始训练，记录开始时间
    begin_time = time()
    # todo 加载数据集， 修改为你的数据集的路径
    train_ds, val_ds, class_names = data_load(r"D:\Python\PythonProject\Truck_identification\Final data\train",
                                              r"D:\Python\PythonProject\Truck_identification\Final data\val", 224, 224, 16)
    print(class_names)
    # 加载模型
    model = model_load(class_num=len(class_names))

    # 指明训练的轮数epoch，开始训练
    history = model.fit(train_ds, validation_data=val_ds, callbacks=callbacks, epochs=epochs)
    # todo 保存模型， 修改为你要保存的模型的名称
    model.save("models/ResNet.h5")
    # 记录结束时间
    end_time = time()
    run_time = end_time - begin_time
    print('该循环程序运行时间：', run_time, "s")
    # 绘制模型训练过程图
    show_loss_acc(history)


if __name__ == '__main__':
    train(epochs=100)
