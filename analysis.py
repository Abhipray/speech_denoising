"""analysis.py

This script runs different speech denoising schemes on a noisy dataset and dumps denoised audio and metrics

Generate the noisy dataset using generate_dataset.py script.
"""
import pathlib
import librosa
import pandas as pd
import numpy as np
import os

np.random.seed(89)  # Make experiment reproducible

from tqdm.autonotebook import tqdm
from ale_anc import AleDenoiser
from tabulate import tabulate

noisy_dataset = pathlib.Path("dataset/speech_plus_noise/")
clean_dataset = pathlib.Path("dataset/speech/")
output_dir = pathlib.Path("out/")
conditions = list(noisy_dataset.iterdir())
print("There are {} conditions".format(len(conditions)))

sr = 16000

os.makedirs(output_dir, exist_ok=True)


def rms_energy(x):
    return 10 * np.log10((1e-12 + x.dot(x)) / len(x))


results = {}

for scheme in tqdm(range(1, 4)):
    condition_results = []
    for condition in tqdm(conditions):
        tqdm.write(condition.stem)
        all = set(condition.glob("*.wav"))
        noise_only = set(condition.glob("*_noise.wav"))
        noisy = all - noise_only

        avg_snr_boost = 0
        for sample in noisy:
            # De-noise the noisy signal
            y, sr = librosa.load(sample, sr)
            ale_denoiser = AleDenoiser(sr, scheme=scheme)
            s_hat, n_hat = ale_denoiser.ale_anc(y)

            # Open the noise only file
            noise, sr = librosa.load(
                str(sample.parent) + str(sample).split(".wav")[-1] + "/" +
                sample.stem + "_noise.wav", sr)

            # Open the signal only file
            speech, sr = librosa.load(clean_dataset / sample.name, sr)

            # Pass speech and noise through the filter and compute output SNR
            n_hat2 = ale_denoiser.feed_forward(noise)
            s_hat2 = ale_denoiser.feed_forward(speech)

            # Write de-noised audio out

            # Ignore the first 10% of the audio file since that part has a loud pop for when the adaptive filter has
            # random weights
            samp_idx = int(0.1 * len(s_hat2))

            os.makedirs(
                output_dir / ("scheme_" + str(scheme)) / condition.stem,
                exist_ok=True)
            librosa.output.write_wav(
                output_dir / ("scheme_" + str(scheme)) / condition.stem /
                sample.name, s_hat[samp_idx:], sr)

            output_snr = rms_energy(s_hat2[samp_idx:]) - rms_energy(
                n_hat2[samp_idx:])
            input_snr = rms_energy(speech[samp_idx:]) - rms_energy(
                noise[samp_idx:])

            avg_snr_boost += (output_snr - input_snr)

        avg_snr_boost /= len(noisy)
        condition_results.append(avg_snr_boost)
    results[scheme] = condition_results

# Display results
col_names = [condition.name for condition in conditions]
df = pd.DataFrame.from_dict(results, orient='index', columns=col_names)
print(
    tabulate(
        df.T,
        tablefmt="github",
        headers=["scheme/condition"] + list(range(1, 4))))
