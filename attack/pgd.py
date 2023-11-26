import torch
import torch.nn as nn
from .attack import Attacker


class PGD(Attacker):
    def __init__(self, task, config) -> None:
        super(PGD, self).__init__(task, config)
        
    def initialize(self, wav, interval=None):
        self.perturb = nn.Parameter(torch.zeros(wav.shape[-1], device=wav.device))
        torch.nn.init.normal_(self.perturb, 0, self.config.epsilon)
        self.optimizer = torch.optim.Adam([self.perturb], lr=self.config.lr)

    def generate(self, wav):
        wav_ = wav + torch.clamp(self.perturb, -self.config.epsilon, self.config.epsilon)
        wav_ = torch.clamp(wav_, -1, 1)
        return wav_
    
    def penalty(self, wav, wav_):
        return 0
    
    def __str__(self) -> str:
        return 'PGD-' + str(self.config.epsilon)