from torch import nn
import torch
from torchvision.models import efficientnet_b0
import torch.nn.functional as F

from utils.dann_utils import GradReverse


class Feature_extractor(nn.Module):
    def __init__(self):
        super(Feature_extractor, self).__init__()
        # 虽然这里用不到model的最后分类层，但还是将其类别按2来处理
        model = efficientnet_b0(weights=None, num_classes=2)
        self.encoder = nn.Sequential(
            model.features,
            model.avgpool,
            nn.Flatten(),     # 这里添加flattern是为了使features传入classifier不尺寸错误
        )


        # 模型初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        x = self.encoder(x)
        # x = torch.flatten(x, 1)
        # x = F.relu(x)
        return x


class Label_classifier(nn.Module):
    def __init__(self, num_classes=2):
        super(Label_classifier, self).__init__()
        model = efficientnet_b0(weights=None, num_classes=num_classes)
        self.label_classifier = model.classifier

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        x = self.label_classifier(x)
        return x


class Domain_Classifier(nn.Module):
    def __init__(self, num_classes=2):
        super(Domain_Classifier, self).__init__()

        self.domain_classifier = nn.Sequential(
            nn.Linear(1280, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_classes),
        )

        # 模型初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, x, alpha=-1):
        # GRL是DANN的核心点之一，不能漏掉
        x = GradReverse.apply(x, alpha)
        x = self.domain_classifier(x)
        # x = torch.sigmoid(x)      # 暂时不加这一层，并将loss改为交叉熵损失
        return x



if __name__ == '__main__':
    rand_input = torch.ones(size=(6, 3, 224, 224))
    # model = Feature_extractor()
    # label_cls = Label_classifier()
    # domain_cls = Domain_Classifier()
    #
    # features = model(rand_input)
    # label_out = label_cls(features)
    # domain_out = domain_cls(features, 0.05)
    #
    # print('label_out', label_out.shape)
    # print('domain_out', domain_out.shape)

    # ---
    from torchsummary import summary

    # model = efficientnet_b0(weights=None, num_classes=2)
    # summary(model, (3, 224, 224))

    m1 = Feature_extractor()
    m2 = Label_classifier()
    m3 = Domain_Classifier()

    # print(m3)

    features = m1(rand_input)
    # label = m2(features)
    # domain = m3(features)
    #
    print(features.shape)
    # print(label.shape)
    # print(domain.shape)










































