import torch 
import torchvision
import torchvision.transforms as transforms

class CIFAR10:
    # 60000 32x32 RGB images in 10 classes, with 6000 images per class
    # train set = 50,000 images
    # 50,000 / 4 = 12500 batch
    # each batch contains 4 images
    # test set = 10,000 images
    # 10,000 / 4 = 2500 batch
    def __init__(self):
        # data augmentation
        self.transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )
        self.trainloader = None 
        self.testloader = None
        self.batch_size = 4
        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    def generate(self):
        # train dataset
        trainset = torchvision.datasets.CIFAR10(root="./data", train=True, 
                                                download=True, transform=self.transform)
        self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size, 
                                                       shuffle=True, num_workers=2)

        # test dataset
        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=self.transform)
        self.testloader = torch.utils.data.DataLoader(testset, batch_size=self.batch_size,
                                         shuffle=False, num_workers=2)
