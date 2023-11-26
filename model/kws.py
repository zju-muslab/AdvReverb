import torch
import torch.nn as nn
from torchaudio.transforms import MelSpectrogram
from .bc_resnet import BCResNet


class LogMelSpec(nn.Module):
    
    def __init__(self, sr, n_mels, n_fft, win_length, hop_length):
        super().__init__()
        self.melspec = MelSpectrogram(
            sample_rate=sr,
            win_length=int(win_length*sr/1000),
            hop_length=int(hop_length*sr/1000),
            n_fft=n_fft,
            n_mels=n_mels
        )
    
    def compression(self, x, C=1, clip_val=1e-5):
        return torch.log(torch.clamp(x, min=clip_val) * C)
    
    def forward(self, x):
        return self.compression(self.melspec(x))


class KWS(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.feature, self.name, self.n_class = config.feature, config.name, config.n_class 
        self.filter = LogMelSpec(sr=self.feature.sr, n_mels=self.feature.n_mels, n_fft=self.feature.n_fft, win_length=self.feature.win_length, hop_length=self.feature.hop_length)
        self.extractor = BCResNet(config.network)
        hid_dim = config.network.head_layer.hid_dim
        self.classifier = nn.Linear(hid_dim, self.n_class)
            
    
    def forward(self, wavs):
        fea = self.filter(wavs)
        emb = self.extractor(fea)
        out_prob = self.classifier(emb)
        score, index = out_prob.max(dim=-1)
        if self.training:
            return out_prob, score, index, emb
        else:
            return out_prob, index

