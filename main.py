from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import time
from torch.autograd import Variable

transform = transforms.Compose(
    [
        transforms.Scale([64, 64]),
        transforms.ToTensor()
    ]
)
# 如果不存在DogsVSCats文件夹，请运行./data/DogsVSCtas.py
train_dataset = datasets.ImageFolder(root='./dataset/DogsVSCats/train/', transform=transform)
test_dataset = datasets.ImageFolder(root='./dataset/DogsVSCats/valid/', transform=transform)

train_dataloader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=16, shuffle=True)


class Models(torch.nn.Module):
    def __init__(self):
        super(Models, self).__init__()
        self.Conv = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),

            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),

            torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),

            torch.nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.Classes = torch.nn.Sequential(
            torch.nn.Linear(4 * 4 * 512, 1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(1024, 1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(1024, 2)
        )

    def forward(self, input):
        x = self.Conv(input)
        x = x.view(-1, 4 * 4 * 512)
        x = self.Classes(x)
        return x


model = Models()
# print(model)

loss_f = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

epoch_n = 10
time_open = time.time()

for epoch in range(epoch_n):
    time_start = time.time()
    print("Epoch{}/{}".format(epoch, epoch_n - 1))
    print("-" * 10)
    print("Training....")
    model.train(True)
    running_loss = 0.0
    running_corrects = 0
    for batch, data in enumerate(train_dataloader, 1):
        X, y = data
        X, y = Variable(X), Variable(y)
        y_pred = model(X)  # 预测结果
        _, pred = torch.max(y_pred.data, 1)
        optimizer.zero_grad()
        loss = loss_f(y_pred, y)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 向前一步
        running_loss += loss.item()  # 累计损失
        running_corrects += torch.sum(pred == y.data)
        if batch % 500 == 0:
            print("Batch{},Train Loss:{:.4f}, Train ACC:{:.4f}".format(batch, running_loss / batch,
                                                                       100 * running_corrects / (16 * batch)))
    epoch_loss = running_loss * 16 / len(train_dataset)
    epoch_acc = 100 * running_corrects / len(train_dataset)
    print("Train Loss:{:.4f} Acc:{:.4f}%".format(epoch_loss, epoch_acc))
    print("Train time:{}".format(time.time() - time_start))

    time_start = time.time()
    print("Validing...")
    model.train(False)
    running_loss = 0.0
    running_corrects = 0
    for batch, data in enumerate(test_dataloader, 1):
        X, y = data
        X, y = Variable(X), Variable(y)
        y_pred = model(X)  # 预测结果
        _, pred = torch.max(y_pred.data, 1)
        optimizer.zero_grad()
        loss = loss_f(y_pred, y)  # 计算损失
        running_loss += loss.item()  # 累计损失
        running_corrects += torch.sum(pred == y.data)
        if batch % 500 == 0:
            print("Batch{},Train Loss:{:.4f}, Train ACC:{:.4f}".format(batch, running_loss / batch,
                                                                       100 * running_corrects / (16 * batch)))
    epoch_loss = running_loss * 16 / len(test_dataset)
    epoch_acc = 100 * running_corrects / len(test_dataset)
    print("Valid Loss:{:.4f} Acc:{:.4f}%".format(epoch_loss, epoch_acc))
    print("Valid time:{}".format(time.time() - time_start))

time_end = time.time() - time_open
print(time_end)
