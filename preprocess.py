import argparse
from pathlib import Path
import json
import shutil

from tqdm import tqdm

import torch
import torchaudio
from torchaudio.functional import resample

from module.utils.config import load_json_file
from module.utils.f0_estimation import estimate_f0
from module.utils.mel import LogMelSpectrogram


def preprocess(input_path: Path, config):
    frame_size = config["preprocess"]["frame_size"]
    length = config['preprocess']['length']
    sample_rate = config["preprocess"]["sample_rate"]
    pe_algorithm = config['preprocess']['pitch_estimation']
    n_mels = config['preprocess']['n_mels']
    n_fft = config['preprocess']['n_fft']
    cache_path = Path(config['data_module']['cache_dir'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    to_mel = LogMelSpectrogram(sample_rate, n_fft, frame_size, n_mels)
    counter = 0

    if not cache_path.exists():
        cache_path.mkdir()

    for waveform_path in tqdm(list(input_path.rglob("*.wav"))):
        # load waveform
        wf, sr = torchaudio.load(waveform_path)

        # resampling
        if sr != sample_rate:
            wf = resample(wf, sr, sample_rate)
            torchaudio.save(waveform_path, wf, sample_rate)

        # mix down
        wf = wf.mean(dim=0, keepdim=True) # [1, length]

        # split
        segments = wf.split(length, dim=1)
        for segment in segments:
            if segment.shape[1] < length:
                segment = torch.cat([segment, torch.zeros(1, length - segment.shape[1])], dim=1)
            torchaudio.save(cache_path / f"{counter}.wav", segment, sample_rate)
            f0 = estimate_f0(segment.to(device), sample_rate, frame_size, algorithm=pe_algorithm).cpu()
            mel = to_mel(segment)
            d = {
                "f0": f0,
                "mel": mel
            }
            torch.save(d, cache_path / f"{counter}.pt")
            counter += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str)
    parser.add_argument("-c", "--config", type=str, default="config/default.json")
    args = parser.parse_args()
    config = load_json_file(args.config)
    
    preprocess(Path(args.input), config)


