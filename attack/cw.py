import torch
import torch.nn as nn
import torch.nn.functional as F
from .attack import Attacker


class CW(Attacker):
    def __init__(self, task, config) -> None:
        super(CW, self).__init__(task, config)
    
    def initialize(self, wav, interval=None):
        self.w = nn.Parameter(torch.zeros(wav.shape[-1], device=wav.device))
        torch.nn.init.normal_(self.w, 0, 0.1)
        self.optimizer = torch.optim.Adam([self.w], lr=self.config.lr)
    
    def generate(self, wav):
        wav_ = torch.tanh(self.w)
        wav_ = wav_.reshape_as(wav)
        return wav_
    
    def penalty(self, wav, wav_):
        return F.mse_loss(wav, wav_, reduction='sum')
        
    def __str__(self) -> str:
        return 'CW-' + str(self.config.alpha)