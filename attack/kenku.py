import torch
import torch.nn as nn
import torch.nn.functional as F
from .attack import Attacker
from torchaudio.transforms import MFCC
import torchaudio as ta

class KENKU(Attacker):
    
    def __init__(self, task, config) -> None:
        super(KENKU, self).__init__(task, config)
        self.mfcc = MFCC(n_mfcc=self.config.n_mfcc, 
                         melkwargs={
                            "n_mels": self.config.n_mels,
                            "n_fft": self.config.n_fft,
                            "hop_length": self.config.hop_length
                         }
                        )
        self.tgt_wav, _ = ta.load(self.config.tgt_wav_path)

    
    def initialize(self, wav, interval=None):
        self.perturb = nn.Parameter(torch.zeros(wav.shape[-1], device=wav.device))
        torch.nn.init.uniform_(self.perturb, -self.config.epsilon, self.config.epsilon)
        self.optimizer = torch.optim.Adam([self.perturb], lr=self.config.lr)
        self.mfcc.to(wav.device)
        self.tgt_feat = self.mfcc(self.tgt_wav.to(wav.device))
    
    def generate(self, wav):
        wav_ = wav + self.perturb
        wav_ = torch.clamp(wav_, -1, 1)
        return wav_
    
    def penalty(self, wav, wav_):
        return F.mse_loss(wav, wav_, reduction='sum')
    
    def acoustic_loss(self, adv_wav):
        adv_feat = self.mfcc(adv_wav)
        return F.mse_loss(adv_feat, self.tgt_feat)
        
    def __str__(self) -> str:
        return 'KENKU-' + str(self.config.alpha)