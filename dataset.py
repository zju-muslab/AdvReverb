# %%
import os
import torch
import numpy as np
import pandas as pd
import torchaudio as ta
from torch.utils.data import Dataset
from ast import literal_eval


class WavDataset(Dataset):
    
    def __init__(self, src_path, wav_len=None):
        self.wav_len = wav_len
        if os.path.exists(src_path):
            df = pd.read_csv(src_path)
            self.fpaths = df['path']
            self.labels = df['label']
            self.interval = df['interval'].apply(literal_eval)
            self.datas, self.durations = [], []
            for fpath in self.fpaths:
                wav, sr = ta.load(fpath)
                self.durations.append(wav.shape[-1]/sr)
                if self.wav_len is not None and wav.shape[-1] < self.wav_len:
                    wav = torch.hstack((wav, torch.zeros(1, self.wav_len-wav.shape[-1])))
                self.datas.append(wav)
            print(f'Dataset: {src_path} | #class {len(np.unique(self.labels))}, #sample {len(self.labels)}, dur {min(self.durations)}~{max(self.durations)}s')
        else:
            raise Exception('No such a directory ' + src_path)
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        wav = self.datas[index]
        if self.wav_len is not None:
            if wav.shape[-1] > self.wav_len:
                start = np.random.randint(0, wav.shape[-1] - self.wav_len)
                wav = wav[:, start:start+self.wav_len]
        return wav, self.labels[index], self.interval[index]


class MusicDataset(Dataset):
    
    def __init__(self, src_path, rir_len=3200, wav_len=None):
        self.wav_len = wav_len
        self.interval = [(i, min(i + rir_len, wav_len), 'p') for i in range(0, wav_len - rir_len, rir_len)]
        if os.path.exists(src_path):
            self.datas, self.durations, self.classes = [], [], []
            for fpath in os.listdir(src_path):
                if fpath.endswith('.wav') or fpath.endswith('.flac') or fpath.endswith('.mp3'):
                    wav, sr = ta.load(os.path.join(src_path, fpath))
                    self.durations.append(wav.shape[-1]/sr)
                    wav = ta.functional.resample(wav[0], sr, 16000)
                    segs = [wav[i:i+wav_len].unsqueeze(0) for i in range(0, wav.shape[-1], wav_len)]
                    segs = list(filter(lambda x: x.abs().max() > 0.2, segs))
                    self.datas += segs
                    self.classes += [fpath] * len(segs)
            print(f'Dataset: {src_path} | #class {len(self.durations)}, #sample {len(self.datas)}, dur {min(self.durations)}~{max(self.durations)}s')
        else:
            raise Exception('No such a directory ' + src_path)
        
    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        return self.datas[index], self.classes[index], self.interval