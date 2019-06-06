"""ale_anc.py
This script implements three different schemes for denoising speech corrupted with additive noise. The three schemes rely on an adaptive filter configure to do linear prediction
"""

import padasip as pa
import numpy as np


def ale_anc(x, fs, l=256, delta_ms=256, scheme=1):
    """
    Scheme 1 (only ale), Scheme 2 (ale+anc using only speech estimate), Scheme 3(ale+anc using speech+noise as primary)
    :param fs: Sample rate of the audio file
    :param x: input noisy speech signal 
    :param l: length of the adaptive predictor
    :param delta_ms: the number of samples into the future the linear predictor is predicting. 
    Set it to higher thant correlation range for the noise signal but less than the correlation range for the speech signal. 
    :param scheme: {1, 2 or 3} See description above.
    :return: s_hat, n_hat, weights, d_ale
    """
    delta = int(delta_ms * fs / 1000)

    x_ale = pa.input_from_history(np.concatenate((np.zeros(delta, ), x)), l)

    # Create desired signal
    d_ale = np.zeros((x_ale.shape[0], ))
    for i in range(delta, x_ale.shape[0]):
        d_ale[i - delta] = x_ale[i][0]

    # ale
    f = pa.filters.FilterNLMS(n=l, mu=2, w="random")
    s_hat, n_hat, weights = f.run(d_ale, x_ale)

    # anc; delay primary input by L/2 samples to allow prediction filter to have two sided impulse response
    if scheme == 2 or scheme == 3:
        f2 = pa.filters.FilterNLMS(n=l, mu=0.1, w="random")
        x_anc = pa.input_from_history(n_hat, l)
        # Delay d_anc by N/2
        if scheme == 2:
            d_anc = s_hat[:x_anc.shape[0] - l // 2]
        else:
            d_anc = x[:x_anc.shape[0] - l // 2]
        d_anc = np.concatenate([np.zeros(l // 2), d_anc])
        x_anc = x_anc[:len(d_anc), :]
        print(d_anc.shape, x_anc.shape)
        n_hat, s_hat, weights = f2.run(d_anc, x_anc)

    return s_hat, n_hat, weights, d_ale
