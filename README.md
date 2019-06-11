This repo implements a speech denoiser based on linear adaptive filtering. The denoiser is based on linear predictor that can estimate the narrow-band signals (eg. speech) from wide-band interference (eg. white noise). 

There are three denoising schemes implemented:

![predictive denoiser](predictive_denoiser.png)

## Experimental Results

Six speech files were corrupted with white noise and pink noise at four different SNR levels: [-10dB, -3dB, 0dB, 3dB, 10dB]

The results are captured in the table below. The numbers in the table are the SNR boost from the de-noiser. It is measured by passing the speech only and noise only signals through the same path (through the learned adaptive filter weights); the output SNR is then computed from the power of the two output signals. This works because of the linearity of the adaptive filters.

| scheme/condition   |         1 |         2 |         3 |
|--------------------|-----------|-----------|-----------|
| pink_-10dB         |  4.13678  |  1.68607  |  1.68693  |
| white_0dB          |  3.38535  |  2.15242  |  2.27038  |
| white_3dB          |  4.24325  |  1.98232  |  2.00768  |
| pink_10dB          | -3.00616  | -5.74865  | -5.64084  |
| white_-3dB         |  4.06389  |  4.12627  |  4.08265  |
| pink_0dB           |  1.33527  |  0.986568 |  1.01922  |
| white_-10dB        |  1.98016  | -4.22426  | -2.34385  |
| white_10dB         |  3.64783  | -0.6496   | -0.693271 |
| pink_-3dB          |  2.79054  |  3.10954  |  3.1155   |
| pink_3dB           | -0.608852 | -1.59065  | -1.58434  |

The results were generated with the following configuration. See analysis.py for documentation of what these parameters mean:

|param     | value   |
|----------|---------|
| fs       | 16000   |
| l1       |   256   |
| l2       |   256   |
| mu1      |     2   |
| mu2      |     0.5 |
| delta_ms |    16   |

 To listen to the output of the denoiser under these different conditions, see the [out](/out) folder. 
 
 
This work was done as part of Stanford's EE373A: Adaptive signal processing with Dr. Widrow.