import torch.nn as nn


class Attacker(nn.Module):
    
    def __init__(self, task, config) -> None:
        super().__init__()
        self.config = config
        self.task = task
        
    def initialize(self, wav, interval=None):
        pass
        
    def generate(self, wav):
        pass
    
    def penalty(self, wav, wav_):
        pass
    
    def __str__(self) -> str:
        pass