from torchvision import datasets
from torchvision import transforms

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

train_transform = transforms.Compose([
    transforms.ToTensor(),
    normalize
])

#train_dataset = datasets.CIFAR10(root="./data/", train=True,
#                                 download=False, transform=train_transform)
train_dataset = datasets.CIFAR10(root="./data/", train=False,
                                 download=True, transform=train_transform)
print(dir(train_dataset))
quit()
for i, data in enumerate(train_dataset):
    print(i)
    print(data)
    print(dir(data))
    if i > 5:
        break
