"""
VGG
"""
import torch
from torch import nn

from init_weight import init_weights_


class VGG_A(nn.Module):
    def __init__(self, inp_ch=3, num_classes=10):
        super().__init__()

        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels=inp_ch, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.stage2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.stage3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.stage4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.stage5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.stage6 = nn.Sequential(
            nn.Linear(512*1*1, 512),
            # nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            # nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, num_classes))


    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.stage6(x.view(-1, 512*1*1))
        return x


class VGG_A_Batch(nn.Module):
    def __init__(self, inp_ch=3, num_classes=10):
        super().__init__()

        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels=inp_ch, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.stage2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.stage3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.stage4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.stage5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.stage6 = nn.Sequential(
            nn.Linear(512*1*1, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, num_classes))


    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.stage6(x.view(-1, 512*1*1))
        return x


class VGG_A_Batch_Noisy(nn.Module):
    def __init__(self, inp_ch=3, num_classes=10, n_mu=0.5, n_sigma=1.25,
                 r_mu=0.1, r_sigma=0.1, device='cuda:0'):
        self.n_mu = n_mu
        self.n_sigma = n_sigma
        self.r_mu = r_mu
        self.r_sigma = r_sigma
        self.device = device
        super().__init__()
        self.maxPool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels=inp_ch, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU())

        self.stage2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU())

        self.stage3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())

        self.stage4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())

        self.stage5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())

        self.stage6 = nn.Sequential(
            nn.Linear(512*1*1, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, num_classes))


    def forward(self, x):
        mu_t = torch.FloatTensor(1).uniform_(-self.n_mu, self.n_mu).item()
        sigma_t = torch.FloatTensor(1).uniform_(1, self.n_sigma).item()
        x = self.stage1(x)
        m_t = torch.FloatTensor(x.shape[0], x.shape[1], x.shape[2], x.shape[3])\
            .uniform_(mu_t - self.r_mu, mu_t + self.r_mu).to(self.device)
        s_t = torch.FloatTensor(x.shape[0], x.shape[1], x.shape[2], x.shape[3])\
            .normal_(sigma_t, self.r_sigma).to(self.device)
        x = x * s_t + m_t
        x = self.maxPool(x)
        x = self.stage2(x)
        m_t = torch.FloatTensor(x.shape[0], x.shape[1], x.shape[2], x.shape[3]) \
            .uniform_(mu_t - self.r_mu, mu_t + self.r_mu).to(self.device)
        s_t = torch.FloatTensor(x.shape[0], x.shape[1], x.shape[2], x.shape[3]) \
            .normal_(sigma_t, self.r_sigma).to(self.device)
        x = x * s_t + m_t
        x = self.maxPool(x)
        x = self.stage3(x)
        m_t = torch.FloatTensor(x.shape[0], x.shape[1], x.shape[2], x.shape[3]) \
            .uniform_(mu_t - self.r_mu, mu_t + self.r_mu).to(self.device)
        s_t = torch.FloatTensor(x.shape[0], x.shape[1], x.shape[2], x.shape[3]) \
            .normal_(sigma_t, self.r_sigma).to(self.device)
        x = x * s_t + m_t
        x = self.maxPool(x)
        x = self.stage4(x)
        m_t = torch.FloatTensor(x.shape[0], x.shape[1], x.shape[2], x.shape[3]) \
            .uniform_(mu_t - self.r_mu, mu_t + self.r_mu).to(self.device)
        s_t = torch.FloatTensor(x.shape[0], x.shape[1], x.shape[2], x.shape[3]) \
            .normal_(sigma_t, self.r_sigma).to(self.device)
        x = x * s_t + m_t
        x = self.maxPool(x)
        x = self.stage5(x)
        m_t = torch.FloatTensor(x.shape[0], x.shape[1], x.shape[2], x.shape[3]) \
            .uniform_(mu_t - self.r_mu, mu_t + self.r_mu).to(self.device)
        s_t = torch.FloatTensor(x.shape[0], x.shape[1], x.shape[2], x.shape[3]) \
            .normal_(sigma_t, self.r_sigma).to(self.device)
        x = x * s_t + m_t
        x = self.maxPool(x)
        x = self.stage6(x.view(-1, 512*1*1))
        return x


class VGG_A_Layer(nn.Module):
    def __init__(self,shape, inp_ch=3, num_classes=10):
        super().__init__()
        self.shape = shape

        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels=inp_ch, out_channels=64, kernel_size=3, padding=1),
            nn.LayerNorm([64] + list(self.shape[1:])),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.stage2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.LayerNorm([128, self.shape[1] // 2, self.shape[2] // 2]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.stage3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.LayerNorm([256, self.shape[1] // 4, self.shape[2] // 4]),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.LayerNorm([256, self.shape[1] // 4, self.shape[2] // 4]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.stage4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.LayerNorm([512, self.shape[1] // 8, self.shape[2] // 8]),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.LayerNorm([512, self.shape[1] // 8, self.shape[2] // 8]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.stage5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.LayerNorm([512, self.shape[1] // 16, self.shape[2] // 16]),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.LayerNorm([512, self.shape[1] // 16, self.shape[2] // 16]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.stage6 = nn.Sequential(
            nn.Linear(512*1*1, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, num_classes))


    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.stage6(x.view(-1, 512*1*1))
        return x


class VGGA(nn.Module):

    def __init__(self, inp_ch=3, num_classes=10, init_weights=True):
        super().__init__()

        self.features = nn.Sequential(
            # stage 1
            nn.Conv2d(in_channels=inp_ch, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # stage 2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # stage 3
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # stage 4
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # stage5
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.classifier = nn.Sequential(
            nn.Linear(512 * 1 * 1, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes))

        if init_weights:
            self._init_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(-1, 512 * 1 * 1))
        return x

    def _init_weights(self):
        for m in self.modules():
            init_weights_(m)


class VGGABatchNorm(nn.Module):

    def __init__(self, inp_ch=3, num_classes=10, init_weights=True):
        super().__init__()

        self.features = nn.Sequential(
            # stage 1
            nn.Conv2d(in_channels=inp_ch, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # stage 2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            #stage 3
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # stage 4
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # stage5
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.classifier = nn.Sequential(
            nn.Linear(512 * 1 * 1, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, num_classes))

        if init_weights:
            self._init_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(-1, 512 * 1 * 1))
        return x

    def _init_weights(self):
        for m in self.modules():
            init_weights_(m)