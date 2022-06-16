import time

time.sleep(4)
print("""
home/ubuntu-1/ASR/espnet/espnet2/layers/stft.py:164: UserWarning: __floordiv__ is deprecated, and its behavior will change in a future version of pytorch. It currently rounds toward 0 (like the 'trunc' function NOT 'floor'). This results in incorrect rounding for negative values. To keep the current behavior, use torch.div(a, b, rounding_mode='trunc'), or for actual floor division, use torch.div(a, b, rounding_mode='floor').
  olens = (ilens - self.n_fft) // self.hop_length + 1
Preprocess(
  (frontend): DefaultFrontend(
    (stft): Stft(n_fft=512, win_length=400, hop_length=160, center=True, normalized=False, onesided=True)
    (frontend): Frontend()
    (logmel): LogMel(sr=16000, n_fft=512, n_mels=80, fmin=0, fmax=8000.0, htk=False)
  )
  (specaug): SpecAug(
    (time_warp): TimeWarp(window=80, mode=bicubic)
    (freq_mask): MaskAlongAxis(mask_width_range=(0, 30), num_mask=2, axis=freq)
    (time_mask): MaskAlongAxis(mask_width_range=(0, 40), num_mask=2, axis=time)
  )
)
""")

print("Generated:  torch.Size([1, 6447, 80])")