# coding=gbk
import torch
from torch import nn, optim
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from hw5z1j_1 import RestNet18


print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
#  ��CIFAR-10 ���ݼ�����ʵ��

def main():
    batchsize = 128


    # �������ݼ�
    # ѵ����
    cifar_train = datasets.CIFAR10('cifar', True,
                                   transform=transforms.Compose([
                                       transforms.Resize((32, 32)),
                                       transforms.ToTensor(),
                                       transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                            std=[0.229, 0.224, 0.225])
                                   ]), download=True)
    # ���Լ�
    cifar_test = datasets.CIFAR10('cifar', False,
                                  transform=transforms.Compose([
                                      transforms.Resize((32, 32)),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                           std=[0.229, 0.224, 0.225])
                                  ]), download=True)

    # �������ݣ���֮ǰ��minist����
    cifar_train = DataLoader(cifar_train, batch_size=batchsize,
                             shuffle=True)
    cifar_test = DataLoader(cifar_test, batch_size=batchsize,
                            shuffle=True)

    x, label = iter(cifar_train).next()
    print('x:', x.shape, 'label:', label.shape)

    device = torch.device('cuda')

    # ʹ�÷�װResNet
    model = RestNet18().to(device)

    # ��ʧ�������Ż���
    criteon = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    print(model)

    # ѵ������100��
    for epoch in range(100):
        model.train()
        for batchidx, (x, label) in enumerate(cifar_train):
            x, label = x.to(device), label.to(device)
            logits = model(x)
            loss = criteon(logits, label)

            # ���򴫲�
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(epoch, 'loss:', loss.item())

        # ��֤ģ��
        model.eval()
        with torch.no_grad():
            # test
            total_correct = 0
            total_num = 0
            for x, label in cifar_test:
                x, label = x.to(device), label.to(device)
                logits = model(x)
                pred = logits.argmax(dim=1)
            correct = torch.eq(pred, label).float().sum().item()
            total_correct += correct
            total_num += x.size(0)
            # �������
        acc = total_correct / total_num
        print(epoch, 'test acc:', acc)
# ���г���
if __name__ == '__main__':
    main()