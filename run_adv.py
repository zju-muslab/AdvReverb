import os
import logging
import hydra
import time
import torch
import torchaudio as ta
import torch.nn.functional as F
from torch.utils.data import DataLoader
from omegaconf import DictConfig, OmegaConf
from speechbrain.nnet.losses import ctc_loss
from speechbrain.utils.metric_stats import EER

import utils
from dataset import WavDataset, MusicDataset
from attack.rir import RIR
from attack.pgd import PGD
from attack.cw import CW
from attack.fm import FM
from attack.kenku import KENKU


log = logging.getLogger(__name__)


def enroll_users(enroll_dataset_path, wav_len, test_dataset, system_name, system, mvn_emb_path, device):
    enroll_file_path = enroll_dataset_path[:-4] + f'_{system_name}.pth'
    if os.path.exists(enroll_file_path) and os.path.exists(mvn_emb_path):
        speaker_profs, eer, threshold = torch.load(enroll_file_path, map_location=device)
        state = torch.load(mvn_emb_path, map_location=device)
        system.hparams.mean_var_norm_emb.count = state['count']
        system.hparams.mean_var_norm_emb.glob_mean = state['glob_mean']
    else:
        system.mods.train()
        system.hparams.mean_var_norm_emb.count = 0
        system.hparams.mean_var_norm_emb.glob_mean = torch.tensor([0]).to(device)
        speaker_profs, pos_score, neg_score = [], [], []
        enroll_dataset = WavDataset(enroll_dataset_path, wav_len=wav_len)
        wav_batch = torch.vstack([wav for (wav, _, _) in enroll_dataset])
        emb_batch = system.encode_batch(wav_batch.to(device), None, True).squeeze(dim=1)
        speaker_profs = emb_batch.reshape(-1, 3, emb_batch.shape[-1]).mean(dim=1)
        dataLoader = DataLoader(test_dataset, batch_size=64, shuffle=True)
        for (wav, label, _) in dataLoader:
            emb = system.encode_batch(wav.squeeze(dim=1).to(device), None, True).squeeze(dim=1)
            for emb_i, label_i in zip(emb, label):
                scores = F.cosine_similarity(emb_i, speaker_profs).cpu().tolist()
                pos_score.append(scores[label_i])
                neg_score += sorted(scores[:label_i] + scores[label_i+1:], reverse=True)[:10]
        eer, threshold = EER(torch.tensor(pos_score), torch.tensor(neg_score))
        torch.save((speaker_profs, eer, threshold), enroll_file_path)
        torch.save({'count': system.hparams.mean_var_norm_emb.count, 'glob_mean': system.hparams.mean_var_norm_emb.glob_mean}, mvn_emb_path)
        system.mods.eval()
    return speaker_profs, eer, threshold
    

@hydra.main(version_base=None, config_path='config', config_name='run_adv_config')
def main(cfg: DictConfig) -> None:
    # config environment
    log.info('\n' + OmegaConf.to_yaml(cfg)) 
    if cfg.gpu:
        device = torch.device(f'cuda:{cfg.device}')
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')
    utils.set_seed(cfg.seed)
    
    # config attacker
    if cfg.adv.name == 'pgd': attacker = PGD(cfg.task, cfg.adv)
    elif cfg.adv.name == 'cw': attacker = CW(cfg.task, cfg.adv)
    elif cfg.adv.name == 'fm': attacker = FM(cfg.task, cfg.adv)
    elif cfg.adv.name == 'rir': 
        cfg.adv.rir_path = os.path.join(cfg.root_dir, cfg.dataset.rir_template_dir, cfg.adv.rir_type + '.wav')
        attacker = RIR(cfg.task, cfg.adv)
    elif cfg.adv.name == 'kenku': 
        cfg.adv.tgt_wav_path = os.path.join(cfg.root_dir, cfg.dataset.kenku_tgt_wav_path)
        attacker = KENKU(cfg.task, cfg.adv)
    else: raise Exception('Invalid attack.')

    # load speech dataset
    if cfg.carrier == 'speech':
        dataset_path = getattr(cfg.dataset, cfg.task)
        dataset = WavDataset(os.path.join(cfg.root_dir, dataset_path), wav_len=cfg.wav_len)
    else:
        dataset_path = getattr(cfg.dataset, 'music')
        if cfg.wav_len is None and cfg.task == 'asr':
            if cfg.adv.name == 'kenku':
                cfg.wav_len = attacker.tgt_wav.shape[-1]
            else:
                cfg.wav_len = 160000 # default to 10s
        dataset = MusicDataset(os.path.join(cfg.root_dir, dataset_path), wav_len=cfg.wav_len)
    log.info('Dataset loaded.')
    
    # load target system
    system_name, system = utils.load_system(cfg.task, getattr(cfg, cfg.task), cfg.root_dir, device)
    # enroll user profiles
    if cfg.task == 'asi':
        enroll_dataset_path = os.path.join(cfg.root_dir, getattr(cfg.dataset, 'enroll'))
        mvn_emb_path = os.path.join(cfg.root_dir, cfg.asi.savedir, 'mean_var_norm_emb_new.ckpt')
        speaker_profs, eer, threshold = enroll_users(enroll_dataset_path, cfg.wav_len, dataset, system_name, system, mvn_emb_path, device)
        log.info(f'system={system_name} enrolled eer={eer:.4f} threshold={threshold:.4f}')
    log.info('Target system loaded.')
    
    # make output
    if cfg.task == 'asr':
        transcript = cfg.speech_trans[cfg.target]
        seq = [system.tokenizer.encode_sequence(list(word)) + [cfg.asr.blank_index] for word in transcript.split(' ')]
        seq = [c for s in seq for c in s]
        target_output = torch.LongTensor(seq).unsqueeze(0).to(device)
    elif cfg.task == 'asi':
        target_output = speaker_profs[cfg.target]
    else:
        target_output = torch.LongTensor([cfg.target]).to(device)
    
    def adv_loss(adv_wav, target_output):
        if cfg.task == 'asr':
            wav_lens = torch.ones(1, device=adv_wav.device)
            emb = system.encode_batch(adv_wav, wav_lens)
            prob = system.hparams.log_softmax(emb)
            loss = ctc_loss(prob, target_output, wav_lens, wav_lens, 0)
        elif cfg.task == 'asi':
            emb = system.encode_batch(adv_wav, None, True).squeeze(dim=1)
            loss = -F.cosine_similarity(emb, target_output)
        else:
            pred, _ = system(adv_wav)
            loss = F.cross_entropy(pred, target_output)
        return loss
    
    def check_success(adv_wav):
        if cfg.task == 'asr':
            transcript = cfg.speech_trans[cfg.target]
            wav_lens = torch.ones(1, device=device)
            seq, _ = system.transcribe_batch(adv_wav, wav_lens)
            return ''.join(seq).replace(' ', '') == transcript.replace(' ', '')
        elif cfg.task == 'asi':
            emb = system.encode_batch(adv_wav, None, True).squeeze(dim=1)
            scores = F.cosine_similarity(emb, speaker_profs)
            return (scores.argmax().item() == cfg.target and scores[cfg.target].item() > threshold)
        else:
            _, pred = system(adv_wav)
            return pred.item() == cfg.target
    
    # generate adversarial example
    res, total_time = [], []
    for i, (ori_wav, _, interval) in enumerate(dataset):
        ori_wav = ori_wav.to(device)
        attacker.initialize(ori_wav, interval)
        attacker.train()
        best_loss = torch.inf
        count = cfg.patience
        time_cost = 0
        success_wav, success_flag = None, False
        for epoch in range(attacker.config.epoch):
            start_time = time.time()
            adv_wav = attacker.generate(ori_wav)
            if epoch == 0 and cfg.adv.name == 'rir':
                ova_wav = adv_wav.detach().clone()
            if cfg.adv.name == 'kenku':
                loss_adv = attacker.acoustic_loss(adv_wav)
            else:
                loss_adv = adv_loss(adv_wav, target_output)
            if cfg.adv.name == 'rir':
                penalty = attacker.penalty(ova_wav, adv_wav)
            else:
                penalty = attacker.penalty(ori_wav, adv_wav)
            loss = loss_adv + attacker.config.alpha * penalty
            attacker.optimizer.zero_grad()
            loss.backward()
            attacker.optimizer.step()
            end_time = time.time()
            time_cost += end_time - start_time
            flag = check_success(adv_wav)
            log.info(f'[{epoch:-4d}/{attacker.config.epoch:4d}] Loss={loss.item():8.6f} | Adv Loss={loss_adv.item():8.6f} | Penalty={loss.item()-loss_adv.item():8.6f} | Success={flag}')
            # early stopping
            if flag:
                if penalty < best_loss:
                    success_wav = adv_wav.detach().cpu()
                    success_flag = True
                    best_loss = penalty
                    count = cfg.patience
                else: count -= 1
                if count < 0: break
        if success_wav is None: success_wav = adv_wav.detach().cpu()
        log.info(f'[{epoch:-4d}/{attacker.config.epoch:4d}] Loss={loss.item():8.6f} | Adv Loss={loss_adv.item():8.6f} | Penalty={loss.item()-loss_adv.item():8.6f}')
        log.info(f'[Trial-{i:-4d}] ' + ('Success' if success_flag else 'Fail') + f' | Time={time_cost:10.4f}')
        res.append(success_flag)
        total_time.append(time_cost)
        if cfg.save_wav:
            utils.set_dir('wav')
            wav_name = os.path.join('wav', str(i).zfill(3) + '_adv.wav')
            ta.save(wav_name, success_wav, 16000)
            ta.save(wav_name.replace('_adv.wav', '_ori.wav'), ova_wav.detach().cpu() if cfg.adv.name == 'rir' else ori_wav.detach().cpu(), 16000)
    asr = sum(res) / len(res)
    total_time = sum(total_time) / len(total_time)
    log.info(f'[system={system_name} attacker={attacker} target={cfg.target}] ASR={asr:6.4f} | Time={total_time:10.4f}')
    

if __name__ == '__main__':
    main()

# example commands:
# python run_adv.py hydra.job.chdir=True task=kws adv=rir adv.epoch=200 adv.alpha=5 target=1 wav_len=16000 carrier=speech gpu=true device=0
# python run_adv.py hydra.job.chdir=True task=asi adv=rir adv.epoch=1000 adv.alpha=5 target=1 wav_len=48000 carrier=speech gpu=true device=1
# python run_adv.py hydra.job.chdir=True task=asr adv=rir adv.epoch=3000 adv.alpha=0.1 target=1 wav_len=null carrier=speech gpu=true device=2