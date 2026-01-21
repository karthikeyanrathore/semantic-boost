import torch 
import torchvision
import torchvision.transforms as transforms

class CIFAR10:
    def __init__(self):
        # augmentation
        self.transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )
        self.trainloader = None 
        self.testloader = None
        self.valloader = None
        self.batch_size = 4 # NOTE: each 12500 training batch contains 4 images
        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    def generate(self):
        # train dataset
        trainset = torchvision.datasets.CIFAR10(root="./data", train=True, 
                                                download=True, transform=self.transform)
        self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size, 
                                                       shuffle=True, num_workers=2)

        # test dataset
        # split testset into test and validation
        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=self.transform)
        validationset, testset = torch.utils.data.random_split(testset, [0.8, 0.2])
        self.testloader = torch.utils.data.DataLoader(testset, batch_size=self.batch_size,
                                         shuffle=False, num_workers=2)
        self.valloader = torch.utils.data.DataLoader(validationset, batch_size=self.batch_size,
                                                     shuffle=False, num_workers=2)

