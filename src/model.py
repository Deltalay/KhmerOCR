import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from transformers import ViTModel

class BiLSTMCTC(nn.Module):
    def __init__(self, in_dim, hidden=256, num_classes=100):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, num_layers=2, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(2 * hidden, num_classes)
    def forward(self, x):
        y, _ = self.lstm(x)
        return self.fc(y)

class PretrainedOCR(nn.Module):
    def __init__(self, num_classes=100, vit_name='google/vit-base-patch16-224'):
        super().__init__()
        self.resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.resnet.fc = nn.Identity()
        self.vit = ViTModel.from_pretrained(vit_name)
        self.v_dim = self.vit.config.hidden_size
        self.decoder = BiLSTMCTC(2048 + self.v_dim, hidden=256, num_classes=num_classes)

    def _resnet_seq(self, x):
        x = self.resnet.conv1(x); x = self.resnet.bn1(x); x = self.resnet.relu(x); x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x); x = self.resnet.layer2(x); x = self.resnet.layer3(x); x = self.resnet.layer4(x)
        x = x.mean(dim=2)
        x = x.permute(0, 2, 1).contiguous()
        return x  # (B, T_r, 2048)

    def _vit_seq(self, x, t_target):
        out = self.vit(x).last_hidden_state
        out = out[:, 1:, :]
        if out.size(1) != t_target:
            out = F.interpolate(out.permute(0, 2, 1), size=t_target, mode='linear', align_corners=False).permute(0, 2, 1)
        return out  # (B, T_r, v_dim)

    def forward(self, x):
        r_seq = self._resnet_seq(x)
        v_seq = self._vit_seq(x, r_seq.size(1))
        seq = torch.cat([r_seq, v_seq], dim=2)
        return self.decoder(seq)  # (B, T, num_classes)
