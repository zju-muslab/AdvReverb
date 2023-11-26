import torch
import numpy as np
import torchaudio as ta
import torch.nn as nn
from scipy import signal
from .attack import Attacker


class FM(Attacker):
        
    def ath_f(self, f):
        f_ = f * 1e-3
        return 3.64 * torch.pow(f_, -0.8) \
            - 6.5 * torch.exp(-0.6 * torch.pow(f_ - 3.3, 2)) \
            + 1e-3 * torch.pow(f_, 4)
        
    def ath_k(self, k):
        return self.ath_f(k / self.config.n_fft * self.config.sr)
        
    def bark_f(self, f):
        return 13 * torch.arctan(0.76 * f * 1e-3) \
            + 3.5 * torch.arctan(torch.pow(f / 7500., 2))
        
    def bark_k(self, k):
        return self.bark_f(k / self.config.n_fft * self.config.sr)

    def two_slops(self, bark_psd_index):
        delta_m = - 6.025 - 0.275 * bark_psd_index[:, 0]
        Ts = []
        for tone in range(bark_psd_index.shape[0]):
            bark_masker = bark_psd_index[tone, 0]
            dz = self.BARK - bark_masker
            sf = torch.zeros_like(dz)
            index = torch.where(dz > 0)[0][0]
            sf[:index] = 27 * dz[:index]
            sf[index:] = (-27 + 0.37 * max(bark_psd_index[tone,1] - 40, 0)) * dz[index:]
            T = bark_psd_index[tone,1] + delta_m[tone] + sf
            Ts.append(T)
        return torch.vstack(Ts)

    def calc_thresh(self, psd):
        # return ATH
        #local maximum
        masker_index = signal.argrelextrema(psd.cpu().numpy(), np.greater)[0]
        masker_index = torch.from_numpy(masker_index).to(psd.device)
        try:
            #remove boundaries
            if masker_index[0] == 0:
                masker_index = masker_index[1:]
            if masker_index[-1] == len(psd) - 1:
                masker_index = masker_index[:-1]
            #larger than ATH
            masker_index = masker_index[psd[masker_index] > self.ATH[masker_index]]
            #smooth
            psd_k = torch.pow(10, psd[masker_index] / 10.)
            psd_k_prev = torch.pow(10, psd[masker_index - 1] / 10.)
            psd_k_post = torch.pow(10, psd[masker_index + 1] / 10.)
            psd_m = 10 * torch.log10(psd_k_prev + psd_k + psd_k_post)
            #local maximum with [-0.5Bark, 0.5Bark]
            bark_m = self.BARK[masker_index]
            bark_psd_index = torch.vstack([bark_m, psd_m, masker_index]).T
            cur, next = 0, 1
            while next < bark_psd_index.shape[0]:
                if next >= bark_psd_index.shape[0]: break
                if bark_psd_index[cur, 2] == -1: break
                while bark_psd_index[next, 0] - bark_psd_index[cur, 0] < 0.5:
                    if bark_psd_index[next, 1] > bark_psd_index[cur, 1]:
                        bark_psd_index[cur, 2] = -1
                        cur = next
                        next = cur + 1
                    else:
                        bark_psd_index[next, 2] = -1
                        next += 1
                    if next >= bark_psd_index.shape[0]: break
                cur = next
                next = cur + 1  
            bark_psd_index = bark_psd_index[bark_psd_index[:,2] != -1]
            #individual threshold
            Ts = self.two_slops(bark_psd_index)
            #global threshold
            Gs = torch.pow(10, self.ATH / 10.) + torch.sum(torch.pow(10, Ts / 10.), dim=0)
            return 10 * torch.log10(Gs)
        except Exception as err:
            print('[local threshold]', err)
            return self.ATH
        
    def calc_norm_psd(self, wav):
        spec = self.spectrogram(wav)
        psd = self.amp2db(spec).squeeze(0)
        psd_max = psd.max()
        psd = self.config.psd_bound - psd_max + psd
        return psd, psd_max
        
    def generate_threshold(self, wav):
        psd, psd_max = self.calc_norm_psd(wav)
        H = []
        for i in range(psd.shape[1]):
            H.append(self.calc_thresh(psd[:,i]))
        H = torch.vstack(H).T
        return H, psd, psd_max
    
    def __init__(self, task, config) -> None:
        super(FM, self).__init__(task, config)
        self.spectrogram = ta.transforms.Spectrogram(
            n_fft=config.n_fft, win_length=config.win_len, hop_length=config.hop_len, 
            power=2, pad_mode='constant', normalized='frame_length')
        self.amp2db = ta.transforms.AmplitudeToDB()
        self.transform = ta.transforms.Spectrogram(
            n_fft=config.n_fft, win_length=config.win_len, hop_length=config.hop_len,
            power=None, return_complex=False)
        bins = torch.arange(0, config.n_fft // 2 + 1)
        self.BARK = self.bark_k(bins)
        ath = self.ath_k(bins)
        self.ATH = torch.where(self.BARK > 1, ath, torch.FloatTensor([float('-inf')]))

    def initialize(self, wav, interval=None):
        self.BARK, self.ATH = self.BARK.to(wav.device), self.ATH.to(wav.device)
        self.spectrogram.to(wav.device)
        self.amp2db.to(wav.device)
        self.transform.to(wav.device)
        self.H, self.psd, self.psd_max = self.generate_threshold(wav)
        self.perturb = nn.Parameter(torch.zeros(wav.shape[-1], device=wav.device))
        torch.nn.init.normal_(self.perturb, 0, 0.01)
        self.optimizer = torch.optim.Adam([self.perturb], lr=self.config.lr)
        
    def generate(self, wav):
        wav_ = wav + self.perturb
        wav_ = torch.clamp(wav_, -1, 1)
        return wav_
    
    def penalty(self, wav, wav_):
        spec = self.spectrogram(self.perturb)
        psd = self.amp2db(spec).squeeze(0)
        psd = self.config.psd_bound - self.psd_max + psd
        return torch.mean(torch.clamp(psd - self.H, 0))
    
    def __str__(self) -> str:
        return 'FM-' + str(self.config.alpha)