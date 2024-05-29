# coding=gbk
import torch
from torch import nn, optim
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from mynet import HakiNet_8, HakiandRegNet_8, VGG16, HakiNet_4, RegNet_8, MiNet_8
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
import matplotlib.pyplot as plt


def main():
    batchsize = 128
    num_epochs = 32
    lrval = 0.000025

    # 定义转换
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # 你可能需要根据你的实际需求调整尺寸
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 训练集
    train_dataset = datasets.ImageFolder('data/output/train', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True)

    # 测试集
    val_dataset = datasets.ImageFolder('data/output/val', transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=batchsize, shuffle=False)

    x, label = iter(train_loader).next()
    print('x:', x.shape, 'label:', label.shape)

    device = torch.device('cuda')
    print(train_dataset.classes)  # 输出类别名称列表
    print(train_dataset.class_to_idx)  # 输出类别到标签的映射

    vgg = [
        8, 8, 'M',  # Conv Block 1: Output size = 64x64
        16, 16, 'M',  # Conv Block 2: Output size = 32x32
        32, 32, 32, 'M',  # Conv Block 3: Output size = 16x16
        64, 64, 64, 'M',  # Conv Block 4: Output size = 8x8
        64, 64, 64, 'M'  # Conv Block 5: Output size = 4x4
    ]

    # 改网络记得修改！！！！！！
    # 改网络记得修改！！！！！！
    # 改网络记得修改！！！！！！
    model = VGG16(vgg).to(device)

    # 损失函数及优化器
    criteon = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lrval)
    # optimizer = optim.SGD(model.parameters(), lr=lrval, momentum=0.9)
    print(model)

    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    val_precisions = []
    val_recalls = []

    # 训练网络100代
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)
        model.train()

        total_loss = 0
        total_num = 0
        total_correct = 0

        for batchidx, (x, label) in enumerate(train_loader):
            x, label = x.to(device), label.to(device)
            logits = model(x)
            loss = criteon(logits, label)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_num += x.size(0)

            # 计算准确率
            pred = logits.argmax(dim=1)
            correct = torch.eq(pred, label).float().sum().item()
            total_correct += correct

        avg_loss = total_loss / len(train_loader)
        train_accuracy = total_correct / total_num
        train_accuracies.append(train_accuracy)
        print(epoch, 'train loss:', avg_loss)
        print(epoch, 'train acc:', train_accuracy)

        # 验证模型
        model.eval()
        total_val_loss = 0
        total_val_correct = 0
        total_val_num = 0
        all_val_preds = []
        all_val_labels = []

        with torch.no_grad():
            for x, label in val_loader:
                x, label = x.to(device), label.to(device)
                logits = model(x)
                val_loss = criteon(logits, label)
                total_val_loss += val_loss.item()

                pred = logits.argmax(dim=1)
                all_val_preds.extend(pred.cpu().numpy())
                all_val_labels.extend(label.cpu().numpy())

                correct = torch.eq(pred, label).float().sum().item()
                total_val_correct += correct
                total_val_num += x.size(0)

        val_acc = total_val_correct / total_val_num
        val_precision = precision_score(all_val_labels, all_val_preds, average='macro', zero_division=0)
        val_recall = recall_score(all_val_labels, all_val_preds, average='macro', zero_division=0)
        avg_val_loss = total_val_loss / len(val_loader)

        print(epoch, 'val acc:', val_acc)
        print(epoch, 'val precision:', val_precision)
        print(epoch, 'val recall:', val_recall)
        print(epoch, 'val loss:', avg_val_loss)

        # 收集指标
        train_losses.append(avg_loss)
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_acc)
        val_precisions.append(val_precision)
        val_recalls.append(val_recall)

    # 绘制曲线图
    plt.figure(figsize=(12, 12))

    # 绘制训练损失曲线
    plt.subplot(3, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # 绘制训练准确率曲线
    plt.subplot(3, 2, 2)
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # 绘制验证损失曲线
    plt.subplot(3, 2, 3)
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # 绘制验证准确率曲线
    plt.subplot(3, 2, 4)
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # 绘制验证精确率和召回率曲线
    plt.subplot(3, 2, 5)
    plt.plot(val_precisions, label='Validation Precision')
    plt.title('Validation Precision')
    plt.xlabel('Epoch')
    plt.ylabel('Precision')
    plt.legend()

    plt.subplot(3, 2, 6)
    plt.plot(val_recalls, label='Validation Recall')
    plt.title('Validation Recall')
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.legend()

    # 显示图表
    plt.tight_layout()
    plt.show()

    # 改网络记得修改！！！！！！
    # 改网络记得修改！！！！！！
    # 改网络记得修改！！！！！！
    # 保存模型参数
    torch.save(model.state_dict(), 'VGG16BEST2.pth')

    # 改网络记得修改！！！！！！
    # 改网络记得修改！！！！！！
    # 改网络记得修改！！！！！！
    # 以下测试部分，仅做测试时前面训练部分可以注释掉
    model = VGG16(vgg).to(device)
    model.load_state_dict(torch.load('VGG16BEST2.pth'))
    model.eval()
    # 加载测试集
    test_dataset = datasets.ImageFolder('data/output/test', transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batchsize, shuffle=False)
    # 收集所有预测和真实标签
    all_preds = []
    all_labels = []
    total_test_loss = 0
    total_test_correct = 0
    total_test_num = 0

    with torch.no_grad():
        for x, label in test_loader:
            x, label = x.to(device), label.to(device)
            logits = model(x)
            pred = logits.argmax(dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(label.cpu().numpy())
            # 计算损失值并累加
            loss = criteon(logits, label)
            total_test_loss += loss.item()
            # 计算准确率
            correct = torch.eq(pred, label).float().sum().item()
            total_test_correct += correct
            total_test_num += x.size(0)

    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    print(cm)
    # 计算总的测试准确率
    test_accuracy = total_test_correct / total_test_num
    print('Average Test Accuracy:', test_accuracy)
    # 计算平均损失值
    avg_test_loss = total_test_loss / len(test_loader)
    print('Average Test Loss:', avg_test_loss)


# 运行程序
if __name__ == '__main__':
    main()