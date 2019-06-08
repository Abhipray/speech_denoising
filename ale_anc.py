"""ale_anc.py
This script implements three different schemes for denoising speech corrupted with additive noise. The three schemes rely on an adaptive filter configure to do linear prediction
"""

import padasip as pa
import numpy as np
import scipy.signal


class AleDenoiser:
    def __init__(self, fs, scheme, l1=256, l2=256, delta_ms=16):
        self.fs = fs
        self.scheme = scheme
        self.l1 = l1
        self.l2 = l2
        self.W1 = None
        self.W2 = None
        self.delta = int(delta_ms * self.fs / 1000)
        assert (self.scheme in {1, 2, 3})

    def ale_anc(self, x):
        """
        Scheme 1 (only ale), Scheme 2 (ale+anc using only speech estimate), Scheme 3(ale+anc using speech+noise as primary)
        :param fs: Sample rate of the audio file
        :param x: input noisy speech signal
        :param l: length of the adaptive predictor
        :param delta_ms: the number of samples into the future the linear predictor is predicting.
        Set it to higher thant correlation range for the noise signal but less than the correlation range for the speech signal.
        :param scheme: {1, 2 or 3} See description above.
        :return: s_hat, n_hat
        """
        # Create a delayed version of the input signal by filtering through z^(-delta)
        num_coeffs = np.zeros((self.delta + 1, ))
        num_coeffs[self.delta] = 1
        x_delayed = scipy.signal.lfilter(num_coeffs, 1, x)

        # Prepare the input into the ALE
        x_ale = pa.input_from_history(x_delayed, self.l1)

        # Create desired signal
        d_ale = x[:x_ale.shape[0]]

        # ale
        f = pa.filters.FilterNLMS(n=self.l1, mu=1, w="random")
        s_hat, n_hat, self.W1 = f.run(d_ale, x_ale)

        # anc; delay primary input by L/2 samples to allow prediction filter to have two sided impulse response
        if self.scheme == 2 or self.scheme == 3:
            f2 = pa.filters.FilterNLMS(n=self.l2, mu=0.5, w="random")
            x_anc = pa.input_from_history(n_hat, self.l2)
            if self.scheme == 2:
                d_anc = s_hat
            else:
                d_anc = d_ale
            # Delay d_anc by L/2
            num_coeffs = np.zeros((self.l2 // 2 + 1, ))
            num_coeffs[self.l2 // 2] = 1
            d_anc = scipy.signal.lfilter(num_coeffs, 1, d_anc)
            d_anc = d_anc[:x_anc.shape[0]]

            n_hat, s_hat, self.W2 = f2.run(d_anc, x_anc)

        return s_hat, n_hat

    def feed_forward(self, x):
        """Pass the signal x through the learned filter weights"""
        if self.W1 is None or (self.scheme >= 2 and self.W2 is None):
            return None

        # Create a delayed version of the input signal by filtering through z^(-delta)
        num_coeffs = np.zeros((self.delta + 1, ))
        num_coeffs[self.delta] = 1
        x_delayed = scipy.signal.lfilter(num_coeffs, 1, x)

        x_ale = pa.input_from_history(x_delayed, self.l1)

        # Create desired signal
        d_ale = x[:x_ale.shape[0]]

        # ale
        e = []

        y = []
        for i in range(self.W1.shape[0]):
            val = np.dot(self.W1[i], x_ale[i])
            y.append(val)
            e.append(d_ale[i] - val)

        if self.scheme == 1:
            return np.array(y)

        # Feed the error signal into the second filter and set the desired filter as the delayed version of the
        # output of the previous AF
        x_anc = pa.input_from_history(e, self.l2)
        if self.scheme == 2:
            d_anc = y
        else:
            d_anc = d_ale
        # Delay d_anc by L/2
        num_coeffs = np.zeros((self.l2 // 2 + 1, ))
        num_coeffs[self.l2 // 2] = 1
        d_anc = scipy.signal.lfilter(num_coeffs, 1, d_anc)
        d_anc = d_anc[:x_anc.shape[0]]

        e = []
        for i in range(self.W2.shape[0]):
            val = np.dot(self.W2[i], x_anc[i])
            e.append(d_anc[i] - val)
        return np.array(e)
