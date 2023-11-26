import os
import shutil
import random
import torch
import numpy as np
from model.kws import KWS
from speechbrain.pretrained import EncoderClassifier, EncoderASR


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def set_dir(path, recreate=False):
    if not os.path.exists(path):
        os.makedirs(path)
    elif recreate:
        shutil.rmtree(path)
        os.makedirs(path)
        

def snr(wav, wav_, epsilon=0):
    power_ratio = torch.sum(wav ** 2)  / (torch.sum((wav_ - wav) ** 2) + epsilon)
    return 10 * torch.log10(power_ratio + epsilon)


def load_system(task, config, root_dir, device):
    name = config.name
    if task == 'asr':
        system = EncoderASR.from_hparams(
            source=config.source, 
            savedir=os.path.join(root_dir, config.savedir),
            run_opts={"device":device}
        )
        system.mods.eval()
    elif task == 'asi':
        system = EncoderClassifier.from_hparams(
            source=config.source, 
            savedir=os.path.join(root_dir, config.savedir),
            run_opts={"device": device}
        )
        system.mods.eval()
    elif task == 'kws':
        system = KWS(config)
        ckpt_path = os.path.join(root_dir, config.savedir, f'{config.name}.pth')
        state = torch.load(ckpt_path, map_location=device)
        system.load_state_dict(state)
        system.to(device)
        system.eval()
    else:
        raise Exception('Invalid task.')
    return name, system

