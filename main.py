from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose(
    [
        transforms.Scale([64, 64]),
        transforms.ToTensor()
    ]
)
# 如果不存在DogsVSCats文件夹，请运行./data/DogsVSCtas
train_dataset = datasets.ImageFolder(root='./dataset/DogsVSCats/train/', transform=transform)
test_dataset = datasets.ImageFolder(root='./dataset/DogsVSCats/valid/', transform=transform)

train_dataloader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=16, shuffle=True)

X_train, y_train = next(iter(train_dataloader))
print(len(X_train))
print(len(y_train))
