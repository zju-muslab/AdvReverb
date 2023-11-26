import os
import glob
import argparse
import pandas as pd
import torchaudio as ta
from pesq import pesq
from tqdm import tqdm
from pathlib import Path
from mel_cepstral_distance import get_metrics_wavs
from utils import snr

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', type=str, required=True)
parser.add_argument('--csv-path', type=str, required=True)
args = parser.parse_args()

wav_adv_files = sorted(glob.glob(os.path.join(args.data_dir, '*adv.wav')))
wav_ova_files = [f.replace('adv.wav', 'ori.wav') for f in wav_adv_files]
wav_src_files, wav_tgt_files = wav_adv_files, wav_ova_files
min_num = min(len(wav_src_files), len(wav_tgt_files))
wav_src_files, wav_tgt_files = wav_src_files[:min_num], wav_tgt_files[:min_num]
metrics = []
for wav_src_file, wav_tgt_file in tqdm(zip(wav_src_files, wav_tgt_files)):
    wav_src, sr = ta.load(wav_src_file)
    wav_tgt, sr = ta.load(wav_tgt_file)
    try:
        mcd_score, _, _ = get_metrics_wavs(Path(wav_tgt_file), Path(wav_src_file))
        pesq_score = pesq(sr, wav_tgt.squeeze().numpy(), wav_src.squeeze().numpy(), 'wb')
        snr_score = snr(wav_tgt, wav_src).item()
    except Exception as e:
        print(os.path.basename(wav_src_file) + ' failed.')
        print(e)
    else:
        metrics.append([mcd_score, pesq_score, snr_score])
        print(os.path.basename(wav_src_file) + ' | ' + os.path.basename(wav_tgt_file) + f'| pesq={pesq_score:.4f} | mcd={mcd_score:.4f} | snr={snr_score:.4f}')
metrics = pd.DataFrame(metrics, columns=['MCD', 'PESQ', 'SNR'])
metrics.to_csv(args.csv_path)
print(f'PESQ={metrics["PESQ"].mean():.4f} MCD={metrics["MCD"].mean():.4f} SNR={metrics["SNR"].mean():.4f}')
