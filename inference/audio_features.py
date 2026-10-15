#!/usr/bin/env python3
"""Audio loading plus official VoxBlink2 and legacy Kaldi frontends."""

import math
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

try:
    import soundfile as sf
except Exception:
    sf = None

try:
    from scipy.signal import resample_poly
except Exception:
    resample_poly = None

kaldi = None
TORCHAUDIO_ERROR = "not loaded"
TORCHAUDIO_ATTEMPTED = False


def _load_optional_torchaudio_kaldi():
    """Load the legacy Kaldi backend only when a legacy run requests it."""

    global kaldi, TORCHAUDIO_ATTEMPTED, TORCHAUDIO_ERROR
    if TORCHAUDIO_ATTEMPTED:
        return kaldi
    TORCHAUDIO_ATTEMPTED = True
    try:
        from torchaudio.compliance import kaldi as kaldi_module

        kaldi = kaldi_module
        TORCHAUDIO_ERROR = ""
    except Exception as exc:
        kaldi = None
        TORCHAUDIO_ERROR = "{}: {}".format(type(exc).__name__, exc)
    return kaldi


@dataclass(frozen=True)
class FbankConfig:
    sample_rate: int = 16000
    num_mel_bins: int = 80
    frame_length_ms: float = 25.0
    frame_shift_ms: float = 10.0
    dither: float = 0.0
    preemphasis: float = 0.97
    low_frequency: float = 20.0
    high_frequency: float = 0.0
    cmn: bool = True


def _decode_pcm_wave(path: Path) -> Tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        sample_rate = handle.getframerate()
        sample_width = handle.getsampwidth()
        frames = handle.getnframes()
        raw = handle.readframes(frames)

    if sample_width == 1:
        values = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sample_width == 2:
        values = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sample_width == 3:
        bytes_3 = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        values_i32 = (
            bytes_3[:, 0].astype(np.int32)
            | (bytes_3[:, 1].astype(np.int32) << 8)
            | (bytes_3[:, 2].astype(np.int32) << 16)
        )
        values_i32 = (values_i32 ^ 0x800000) - 0x800000
        values = values_i32.astype(np.float32) / 8388608.0
    elif sample_width == 4:
        values = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError("不支持的 PCM 位宽：{} bytes".format(sample_width))

    if channels > 1:
        values = values.reshape(-1, channels).mean(axis=1)
    return values.astype(np.float32, copy=False), sample_rate


def load_audio(
    path: Path, target_sample_rate: int, remove_dc: bool = True
) -> np.ndarray:
    path = Path(path)
    if sf is not None:
        waveform, sample_rate = sf.read(
            str(path), dtype="float32", always_2d=True
        )
        waveform = waveform.mean(axis=1)
    else:
        waveform, sample_rate = _decode_pcm_wave(path)

    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.size == 0:
        raise ValueError("空音频：{}".format(path))
    if not np.isfinite(waveform).all():
        raise ValueError("音频包含 NaN/Inf：{}".format(path))
    if remove_dc:
        waveform = waveform - np.mean(waveform, dtype=np.float64)

    if sample_rate != target_sample_rate:
        waveform = _resample(waveform, sample_rate, target_sample_rate)
    peak = float(np.max(np.abs(waveform)))
    if peak < 1.0e-8:
        raise ValueError("音频近似静音，无法生成有效声纹：{}".format(path))
    return waveform.astype(np.float32, copy=False)


def extract_voxblink2_logmel(
    waveform: np.ndarray, config: FbankConfig
) -> Tuple[np.ndarray, str]:
    """Reproduce the official VoxBlink2 ResNet100 waveform frontend.

    The reference implementation applies max-amplitude normalization and
    first-order pre-emphasis before MelSpectrogram.  The implementation below
    reproduces the torchaudio defaults (Hann window, centered reflection
    padding, HTK mel scale) using only Torch primitives, then applies log
    compression and per-mel utterance mean normalization.  These details
    intentionally differ from Kaldi FBank and are part of the checkpoint's
    learned input distribution.
    """

    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.ndim != 1 or waveform.size == 0:
        raise ValueError("waveform 必须是一维且非空")

    maximum = float(np.max(np.abs(waveform)))
    if not math.isfinite(maximum) or maximum < 1.0e-8:
        raise ValueError("音频近似静音，无法生成有效声纹")
    normalized = waveform / (maximum + 1.0e-4)
    emphasized = np.empty_like(normalized)
    emphasized[0] = normalized[0]
    emphasized[1:] = normalized[1:] - config.preemphasis * normalized[:-1]

    n_fft = 512
    win_length = int(
        round(config.frame_length_ms / 1000.0 * config.sample_rate)
    )
    hop_length = int(
        round(config.frame_shift_ms / 1000.0 * config.sample_rate)
    )
    with torch.inference_mode():
        samples = torch.from_numpy(emphasized)
        window = torch.hann_window(win_length, dtype=samples.dtype)
        spectrum = torch.stft(
            samples,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=True,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        power = spectrum.abs().pow(2.0)
        filterbank = _htk_mel_filterbank(
            sample_rate=config.sample_rate,
            n_fft=n_fft,
            n_mels=config.num_mel_bins,
            dtype=power.dtype,
        )
        mel = torch.matmul(power.transpose(0, 1), filterbank).transpose(0, 1)
        features = torch.log(mel + 1.0e-6)
        features = features - features.mean(dim=1, keepdim=True)
        features = features.transpose(0, 1).contiguous().cpu().numpy()

    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2 or features.shape[1] != config.num_mel_bins:
        raise RuntimeError(
            "VoxBlink2 LogMel 形状错误：{}".format(tuple(features.shape))
        )
    if not np.isfinite(features).all():
        raise RuntimeError("VoxBlink2 LogMel 包含 NaN/Inf")
    return features, "voxblink2_torch_melspectrogram"


def _htk_mel_filterbank(
    sample_rate: int,
    n_fft: int,
    n_mels: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build torchaudio-compatible default HTK Mel filter weights.

    This is the small, dependency-free equivalent of
    ``torchaudio.functional.melscale_fbanks`` with ``f_min=0``,
    ``f_max=sample_rate/2``, ``norm=None`` and ``mel_scale='htk'``.
    Keeping the operations in Torch reproduces torchaudio's float32 rounding
    while avoiding a runtime dependency on a matching torchaudio binary.
    """

    if sample_rate <= 0 or n_fft <= 0 or n_mels <= 0:
        raise ValueError("Mel filterbank 参数必须为正数")
    n_freqs = n_fft // 2 + 1
    all_freqs = torch.linspace(
        0,
        sample_rate // 2,
        n_freqs,
        dtype=dtype,
    )
    mel_min = 0.0
    mel_max = 2595.0 * math.log10(1.0 + (sample_rate / 2.0) / 700.0)
    mel_points = torch.linspace(
        mel_min,
        mel_max,
        n_mels + 2,
        dtype=dtype,
    )
    frequency_points = 700.0 * (
        torch.pow(torch.tensor(10.0, dtype=dtype), mel_points / 2595.0) - 1.0
    )
    frequency_differences = frequency_points[1:] - frequency_points[:-1]
    slopes = frequency_points.unsqueeze(0) - all_freqs.unsqueeze(1)
    down_slopes = (-1.0 * slopes[:, :-2]) / frequency_differences[:-1]
    up_slopes = slopes[:, 2:] / frequency_differences[1:]
    return torch.maximum(
        torch.zeros(1, dtype=dtype),
        torch.minimum(down_slopes, up_slopes),
    )


def _kaldi_mel_scale(frequency: torch.Tensor) -> torch.Tensor:
    return 1127.0 * torch.log1p(frequency / 700.0)


def _kaldi_mel_filterbank(
    sample_rate: int,
    padded_window_size: int,
    num_mel_bins: int,
    low_frequency: float = 20.0,
    high_frequency: float = 0.0,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build Kaldi-compatible triangular Mel filters with no VTLN warp."""

    nyquist = 0.5 * sample_rate
    if high_frequency <= 0.0:
        high_frequency += nyquist
    if not 0.0 <= low_frequency < high_frequency <= nyquist:
        raise ValueError("非法 Kaldi Mel 频率范围")

    mel_low = 1127.0 * math.log(1.0 + low_frequency / 700.0)
    mel_high = 1127.0 * math.log(1.0 + high_frequency / 700.0)
    mel_delta = (mel_high - mel_low) / (num_mel_bins + 1)
    bin_index = torch.arange(num_mel_bins, dtype=dtype).unsqueeze(1)
    left = mel_low + bin_index * mel_delta
    center = mel_low + (bin_index + 1.0) * mel_delta
    right = mel_low + (bin_index + 2.0) * mel_delta

    fft_bin_width = float(sample_rate) / padded_window_size
    frequencies = (
        fft_bin_width
        * torch.arange(padded_window_size // 2, dtype=dtype)
    )
    mel = _kaldi_mel_scale(frequencies).unsqueeze(0)
    up_slope = (mel - left) / (center - left)
    down_slope = (right - mel) / (right - center)
    filters = torch.maximum(
        torch.zeros(1, dtype=dtype),
        torch.minimum(up_slope, down_slope),
    )
    # Kaldi ignores the Nyquist bin.  Pad the final zero column so the
    # filter matrix can multiply an rFFT with N/2+1 bins.
    return torch.nn.functional.pad(filters, (0, 1), value=0.0)


def extract_kaldi_fbank_torch(
    waveform: np.ndarray,
    config: FbankConfig,
) -> Tuple[np.ndarray, str]:
    """Reproduce 3D-Speaker's torchaudio/Kaldi FBank in pure Torch.

    The public CAM++ inference code calls ``torchaudio.compliance.kaldi.fbank``
    with its defaults, ``dither=0`` and utterance mean normalization.  This
    inference-only implementation keeps that numerical contract without a
    runtime torchaudio binary dependency.
    """

    waveform = ensure_minimum_duration(
        np.asarray(waveform, dtype=np.float32),
        config.sample_rate,
        max(config.frame_length_ms / 1000.0, 0.025),
    )
    if waveform.ndim != 1 or waveform.size == 0:
        raise ValueError("waveform 必须是一维且非空")

    window_size = int(
        config.sample_rate * config.frame_length_ms / 1000.0
    )
    window_shift = int(
        config.sample_rate * config.frame_shift_ms / 1000.0
    )
    padded_window_size = 1 << (window_size - 1).bit_length()
    with torch.inference_mode():
        samples = torch.from_numpy(waveform).contiguous()
        frame_count = 1 + (samples.numel() - window_size) // window_shift
        frames = samples.as_strided(
            (frame_count, window_size),
            (window_shift * samples.stride(0), samples.stride(0)),
        )
        frames = frames - frames.mean(dim=1, keepdim=True)
        previous = torch.nn.functional.pad(
            frames.unsqueeze(0), (1, 0), mode="replicate"
        ).squeeze(0)
        frames = frames - config.preemphasis * previous[:, :-1]
        povey_window = torch.hann_window(
            window_size,
            periodic=False,
            dtype=frames.dtype,
        ).pow(0.85)
        frames = frames * povey_window.unsqueeze(0)
        if padded_window_size != window_size:
            frames = torch.nn.functional.pad(
                frames, (0, padded_window_size - window_size), value=0.0
            )
        spectrum = torch.fft.rfft(frames).abs().pow(2.0)
        filterbank = _kaldi_mel_filterbank(
            sample_rate=config.sample_rate,
            padded_window_size=padded_window_size,
            num_mel_bins=config.num_mel_bins,
            low_frequency=config.low_frequency,
            high_frequency=config.high_frequency,
            dtype=spectrum.dtype,
        )
        epsilon = torch.tensor(
            torch.finfo(torch.float32).eps, dtype=spectrum.dtype
        )
        features = torch.matmul(spectrum, filterbank.transpose(0, 1))
        features = torch.maximum(features, epsilon).log()
        if config.cmn:
            features = features - features.mean(dim=0, keepdim=True)
        features = features.contiguous().cpu().numpy()

    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2 or features.shape[1] != config.num_mel_bins:
        raise RuntimeError(
            "CAM++ Kaldi FBank 形状错误：{}".format(tuple(features.shape))
        )
    if not np.isfinite(features).all():
        raise RuntimeError("CAM++ Kaldi FBank 包含 NaN/Inf")
    return features, "kaldi_fbank_torch_exact"


def _resample(waveform: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate <= 0 or target_rate <= 0:
        raise ValueError("非法采样率：{} -> {}".format(source_rate, target_rate))
    if resample_poly is not None:
        divisor = math.gcd(source_rate, target_rate)
        output = resample_poly(
            waveform,
            target_rate // divisor,
            source_rate // divisor,
        )
        return np.asarray(output, dtype=np.float32)

    output_length = max(1, int(round(waveform.size * target_rate / source_rate)))
    old_positions = np.arange(waveform.size, dtype=np.float64) / source_rate
    new_positions = np.arange(output_length, dtype=np.float64) / target_rate
    return np.interp(new_positions, old_positions, waveform).astype(np.float32)


def ensure_minimum_duration(
    waveform: np.ndarray, sample_rate: int, minimum_seconds: float = 0.40
) -> np.ndarray:
    minimum_samples = max(1, int(round(sample_rate * minimum_seconds)))
    if waveform.size >= minimum_samples:
        return waveform
    repetitions = int(math.ceil(minimum_samples / waveform.size))
    return np.tile(waveform, repetitions)[:minimum_samples].astype(np.float32)


def _hz_to_mel(frequency: np.ndarray) -> np.ndarray:
    return 1127.0 * np.log1p(frequency / 700.0)


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * np.expm1(mel / 1127.0)


def _mel_filterbank(config: FbankConfig, fft_size: int) -> np.ndarray:
    high_frequency = (
        config.sample_rate / 2.0
        if config.high_frequency <= 0
        else config.high_frequency
    )
    mel_points = np.linspace(
        _hz_to_mel(np.asarray(config.low_frequency)),
        _hz_to_mel(np.asarray(high_frequency)),
        config.num_mel_bins + 2,
    )
    frequencies = _mel_to_hz(mel_points)
    bins = np.floor((fft_size + 1) * frequencies / config.sample_rate).astype(int)
    bins = np.clip(bins, 0, fft_size // 2)
    filters = np.zeros((config.num_mel_bins, fft_size // 2 + 1), dtype=np.float32)
    for index in range(config.num_mel_bins):
        left, center, right = bins[index : index + 3]
        center = max(center, left + 1)
        right = max(right, center + 1)
        for frequency_bin in range(left, min(center, filters.shape[1])):
            filters[index, frequency_bin] = (frequency_bin - left) / (center - left)
        for frequency_bin in range(center, min(right, filters.shape[1])):
            filters[index, frequency_bin] = (right - frequency_bin) / (right - center)
    return filters


def _numpy_fbank(waveform: np.ndarray, config: FbankConfig) -> np.ndarray:
    frame_length = int(round(config.sample_rate * config.frame_length_ms / 1000.0))
    frame_shift = int(round(config.sample_rate * config.frame_shift_ms / 1000.0))
    waveform = ensure_minimum_duration(
        waveform,
        config.sample_rate,
        max(config.frame_length_ms / 1000.0, 0.40),
    )
    frame_count = 1 + (waveform.size - frame_length) // frame_shift
    starts = np.arange(frame_count, dtype=np.int64) * frame_shift
    indices = starts[:, None] + np.arange(frame_length, dtype=np.int64)[None, :]
    frames = waveform[indices].astype(np.float32, copy=True)
    frames -= frames.mean(axis=1, keepdims=True)
    if config.preemphasis != 0:
        frames[:, 1:] -= config.preemphasis * frames[:, :-1]
        frames[:, 0] *= 1.0 - config.preemphasis
    frames *= np.hamming(frame_length).astype(np.float32)[None, :]
    fft_size = 1 << int(math.ceil(math.log(frame_length, 2)))
    spectrum = np.fft.rfft(frames, n=fft_size, axis=1)
    power = np.square(np.abs(spectrum)).astype(np.float32)
    filters = _mel_filterbank(config, fft_size)
    # Some Accelerate/BLAS builds leak floating-point status flags from FFT
    # kernels and emit spurious matmul warnings.  The explicit finite check
    # below remains authoritative and still fails on a genuinely invalid FBank.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        mel_energy = np.matmul(power, filters.T)
    features = np.log(np.maximum(mel_energy, np.finfo(np.float32).eps))
    return features.astype(np.float32)


def extract_fbank(
    waveform: np.ndarray, config: FbankConfig, backend: str = "auto"
) -> Tuple[np.ndarray, str]:
    backend = backend.strip().lower()
    if backend not in ("auto", "torchaudio", "numpy"):
        raise ValueError("GAME823_FBANK_BACKEND 必须是 auto/torchaudio/numpy")

    selected = backend
    available_kaldi = None
    if selected in ("auto", "torchaudio"):
        available_kaldi = _load_optional_torchaudio_kaldi()
    if selected == "auto":
        selected = "torchaudio" if available_kaldi is not None else "numpy"

    if selected == "torchaudio":
        if available_kaldi is None:
            raise RuntimeError("torchaudio FBank 不可用：{}".format(TORCHAUDIO_ERROR))
        tensor = torch.from_numpy(waveform).unsqueeze(0) * float(1 << 15)
        features = available_kaldi.fbank(
            tensor,
            num_mel_bins=config.num_mel_bins,
            frame_length=config.frame_length_ms,
            frame_shift=config.frame_shift_ms,
            dither=config.dither,
            sample_frequency=config.sample_rate,
            window_type="hamming",
            use_energy=False,
        ).cpu().numpy()
    else:
        features = _numpy_fbank(waveform, config)

    features = np.asarray(features, dtype=np.float32)
    if config.cmn:
        features = features - features.mean(axis=0, keepdims=True)
    if features.ndim != 2 or features.shape[1] != config.num_mel_bins:
        raise RuntimeError("FBank 形状错误：{}".format(tuple(features.shape)))
    if not np.isfinite(features).all():
        raise RuntimeError("FBank 包含 NaN/Inf")
    return features.astype(np.float32, copy=False), selected
