# coding=gbk
import torch
from torch import nn, optim
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from block_net import HakiNet_8


def main():
    batchsize = 128

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
    test_dataset = datasets.ImageFolder('data/output/test', transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batchsize, shuffle=False)

    x, label = iter(train_loader).next()
    print('x:', x.shape, 'label:', label.shape)

    device = torch.device('cuda')

    # 使用封装ResNet
    model = HakiNet_8().to(device)

    # 损失函数及优化器
    criteon = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    print(model)

    # 训练网络100代
    for epoch in range(5):
        model.train()
        for batchidx, (x, label) in enumerate(train_loader):
            x, label = x.to(device), label.to(device)
            logits = model(x)
            loss = criteon(logits, label)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(epoch, 'loss:', loss.item())

        # 验证模型
        model.eval()
        with torch.no_grad():
            # test
            total_correct = 0
            total_num = 0
            for x, label in test_loader:
                x, label = x.to(device), label.to(device)
                logits = model(x)
                pred = logits.argmax(dim=1)
            correct = torch.eq(pred, label).float().sum().item()
            total_correct += correct
            total_num += x.size(0)
            # 输出精度
        acc = total_correct / total_num
        print(epoch, 'test acc:', acc)
# 运行程序
if __name__ == '__main__':
    main()