import torch
import torchaudio as ta
import torch.nn as nn
import torch.nn.functional as F
from .attack import Attacker


class RIR(Attacker):

    def __init__(self, task, config) -> None:
        super(RIR, self).__init__(task, config)
        rir, _ = ta.load(config.rir_path)
        rir = rir[0]
        rir = rir / (torch.norm(rir, p=2) + 1e-10)
        rir = rir[rir.argmax():rir.argmax() + self.config.rir_len]
        self.rir = rir.flip(dims=(0,))
        self.window = torch.hann_window(self.config.overlap * 2)
    
    def initialize(self, wav, interval=None):
        assert interval is not None
        self.interval = list(filter(lambda x: x[2] != '' and x[1] - x[0] >= self.config.rir_len, interval))
        self.rir = self.rir.to(wav.device)
        if self.config.mode == 'local':
            self.perturb = [nn.Parameter(self.rir.clone()) for _ in range(len(self.interval))]
            self.optimizer = torch.optim.Adam(self.perturb, lr=self.config.lr)
        else:
            self.perturb = nn.Parameter(self.rir.clone())
            self.optimizer = torch.optim.Adam([self.perturb], lr=self.config.lr)
        self.window = self.window.to(wav.device)
            
    def normalize(self, x, vmin, vmax):
        return (x - x.min()) / (x.max() - x.min()) * (vmax - vmin) + vmin
    
    def convolution(self, wav, s, e, rir):
        # phoneme-level convolution
        s -= self.config.rir_len - 1 + self.config.overlap
        if s < 0:
            seg = F.pad(wav[:, :e], (abs(s), 0))
        else:
            seg = wav[:, s:e]
        return F.conv1d(seg.unsqueeze(0), rir.unsqueeze(0).unsqueeze(0))
    
    def concatenation(self, segs):
        # overlap-add concatenation
        new_segs = []
        for k in range(0, len(segs)-1):
            segs[k][:, :, -self.config.overlap:] = segs[k][:, :, -self.config.overlap:] * self.window[self.config.overlap:] + \
                                                    segs[k+1][:, :, :self.config.overlap] * self.window[:self.config.overlap]
            new_segs.append(segs[k][:, :, self.config.overlap:])
        new_segs.append(segs[-1][:, :, self.config.overlap:])
        if new_segs[0].shape[-1] >= self.config.overlap:
            new_segs[0][:, :, :self.config.overlap] *= self.window[:self.config.overlap]
        if new_segs[-1].shape[-1] >= self.config.overlap:
            new_segs[-1][:, :, -self.config.overlap:] *= self.window[self.config.overlap:]
        return torch.concat(new_segs, dim=-1)
    
    def generate(self, wav):
        if self.config.mode == 'local':
            wav_ = []
            p = 0
            for i, perturb_i in enumerate(self.perturb):
                s, e = self.interval[i][0], self.interval[i][1]
                if s > p:
                    wav_.append(self.convolution(wav, p, s, self.rir))
                wav_.append(self.convolution(wav, s, e, perturb_i))
                p = e
            if p < wav.shape[-1]:
                wav_.append(self.convolution(wav, p, wav.shape[-1], self.rir))
            wav_ = self.concatenation(wav_).squeeze(0)
        else:
            wav = F.pad(wav, (self.config.rir_len-1, 0)).unsqueeze(0)
            wav_ = F.conv1d(wav, self.perturb.unsqueeze(0).unsqueeze(0)).squeeze(0)
        wav_ = self.normalize(wav_, wav.min(), wav.max())
        return wav_
    
    def penalty(self, wav, wav_):
        if self.config.mode == 'local':
            rir_loss = 0
            for perturb_i in self.perturb:
                rir_loss += F.mse_loss(self.rir, perturb_i, reduction='sum')
            rir_loss /= len(self.perturb)
        else:
            rir_loss = F.mse_loss(self.rir, self.perturb, reduction='sum')
        wav_loss = F.mse_loss(wav, wav_, reduction='sum') / wav.shape[-1] * 16000 
        return wav_loss + 10 * rir_loss
    
    def __str__(self) -> str:
        return f'RIR-{self.config.rir_type}-{self.config.rir_len}-' + str(self.config.alpha)
    
    def get_perturb(self):
        if self.config.mode == 'local':
            perturbs = []
            for perturb_i in self.perturb:
                perturbs.append(perturb_i.detach().cpu().flip(dims=(0,)))
            return perturbs
        else:
            return self.perturb.detach().cpu().flip(dims=(0,))