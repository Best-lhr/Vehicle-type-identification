from matplotlib import pyplot as plt
path='D:/Python/PythonProject/Truck_identification/results/训练记录/LeNet训练记录.txt'
def acc_text(path):
    ls=[]
    f=open(path,'r',encoding='utf-8')
    text=f.read()
    L=text.split('accuracy:')[1::2]
    for i in range(len(L)):
        r=L[i].split(' ')[1]
        ls.append(float(r))
    return ls

def loss_text(path):
    ls=[]
    f=open(path,'r',encoding='utf-8')
    text=f.read()
    L=text.split('loss:')[1::2]
    for i in range(len(L)):
        r=L[i].split(' ')[1]
        ls.append(float(r))
    return ls

def show_loss_acc():
    # 从history中提取模型训练集和验证集准确率信息和误差信息
    acc=acc_text(path)
    loss=loss_text(path)

    # 按照上下结构将图画输出
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy',color='#32CD32')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()), 1])
    plt.title('Training Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss',color='#FF6347')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.title('Training Loss')
    plt.xlabel('epoch')
    plt.show()
    plt.savefig('results/ceshi.png', dpi=100)

show_loss_acc()


