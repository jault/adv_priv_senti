import torch.nn.functional as F
from torch.nn import Conv2d, Dropout, MaxPool2d, Linear, ConvTranspose2d, UpsamplingNearest2d, Module


class BasicCNN(Module):

    def __init__(self, num_classes):
        super(BasicCNN, self).__init__()

        self.pool = MaxPool2d(kernel_size=2, stride=2)
        self.drop = Dropout(0.8)

        self.cv1 = Conv2d(1, 32, kernel_size=3, stride=1)
        self.cv2 = Conv2d(32, 64, kernel_size=3, stride=1)
        self.cv3 = Conv2d(64, 64, kernel_size=3, stride=1)
        self.cv4 = Conv2d(64, 64, kernel_size=3, stride=1)

        self.fc1 = Linear(4224, 256)
        self.out = Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.cv1(x)))
        x = self.pool(F.relu(self.cv2(x)))
        x = self.pool(F.relu(self.cv3(x)))
        x = self.pool(F.relu(self.cv4(x)))

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = self.out(x)

        return x


class BasicAutoEncoder(Module):

    def __init__(self):
        super(BasicAutoEncoder, self).__init__()

        self.pool = MaxPool2d(kernel_size=2, stride=2)
        self.poolt = UpsamplingNearest2d(scale_factor=2)

        self.cv1 = Conv2d(1, 64, kernel_size=4, stride=1)
        self.cv2 = Conv2d(64, 32, kernel_size=3, stride=1)
        self.cv3 = Conv2d(32, 16, kernel_size=3, stride=1)

        self.cv1t = ConvTranspose2d(16, 32, kernel_size=3, stride=1)
        self.cv2t = ConvTranspose2d(32, 64, kernel_size=3, stride=1)
        self.cv3t = ConvTranspose2d(64, 1, kernel_size=4, stride=1)

    def forward(self, x):
        x = F.relu(self.cv1(x))
        x = self.pool(x)
        x = F.relu(self.cv2(x))
        x = self.pool(x)
        x = F.relu(self.cv3(x))

        x = F.relu(self.cv1t(x))
        x = self.poolt(x)
        x = F.relu(self.cv2t(x))
        x = self.poolt(x)
        x = F.relu(self.cv3t(x))

        return x
