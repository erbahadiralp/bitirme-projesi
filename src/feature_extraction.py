# src/feature_extraction.py (Güncellenmiş Hali)

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import welch
import neurokit2 as nk

def get_statistical_features(segment):
    """
    Bir sinyal segmentinden temel zaman domeni istatistiksel özelliklerini çıkarır.
    """
    features = []
    for i in range(segment.shape[1]):
        lead = segment[:, i]
        
        mean = np.mean(lead)
        std = np.std(lead)
        skew = stats.skew(lead)
        kurt = stats.kurtosis(lead)
        rms = np.sqrt(np.mean(lead**2))
        entropy = stats.entropy(np.histogram(lead, bins=100)[0])
        
        diff = np.diff(lead)
        mean_abs_diff = np.mean(np.abs(diff))
        zero_crossings = len(np.where(np.diff(np.sign(lead)))[0])

        features.extend([mean, std, skew, kurt, rms, entropy, mean_abs_diff, zero_crossings])
        
    return features

def get_frequency_domain_features(segment, fs=100):
    """
    Bir sinyal segmentinden frekans domeni özelliklerini çıkarır.
    """
    features = []
    for i in range(segment.shape[1]):
        lead = segment[:, i]
        
        freqs, psd = welch(lead, fs=fs, nperseg=len(lead))
        
        spectral_entropy = stats.entropy(psd)
        
        features.extend([np.mean(psd), np.std(psd), spectral_entropy])
        
    return features

def get_hrv_and_morphological_features(segment, fs=100):
    """
    Neurokit2 kullanarak HRV ve morfolojik özellikleri çıkarır.
    Sadece tek bir kanaldan (genellikle Lead II) hesaplanır.
    """
    lead_ii = segment[:, 1] if segment.shape[1] > 1 else segment[:, 0]
    
        # EKG sinyalini işle ve R tepelerini bul
        processed_ecg, info = nk.ecg_process(lead_ii, sampling_rate=fs)
        r_peaks = info["ECG_R_Peaks"]

        if len(r_peaks) < 2:
        return [0] * 15

        hrv_time = nk.hrv_time(r_peaks, sampling_rate=fs)
        hrv_freq = nk.hrv_frequency(r_peaks, sampling_rate=fs)
        
        _, waves = nk.ecg_delineate(processed_ecg, r_peaks, sampling_rate=fs)

        qrs_duration = np.nanmean(waves['ECG_QRS_Offsets'] - waves['ECG_QRS_Onsets']) / fs
        r_amplitude = np.nanmean(processed_ecg[waves['ECG_R_Peaks']])
        t_amplitude = np.nanmean(processed_ecg[waves['ECG_T_Peaks']])
        p_amplitude = np.nanmean(processed_ecg[waves['ECG_P_Peaks']])
        
        hrv_features = pd.concat([hrv_time, hrv_freq], axis=1).iloc[0].values
        morph_features = np.array([qrs_duration, r_amplitude, t_amplitude, p_amplitude])
        
        all_features = np.concatenate([hrv_features, morph_features])
    all_features = np.nan_to_num(all_features, nan=0.0)

        if len(all_features) < 15:
            all_features = np.pad(all_features, (0, 15 - len(all_features)))
        return all_features[:15]

def extract_features_from_segments(segments, fs=100):
    """
    Tüm segmentler üzerinden döngüye girerek her biri için tüm özellikleri çıkarır.
    """
    all_feature_list = []
    
    for i, segment in enumerate(segments):
        stat_features = get_statistical_features(segment)
        freq_features = get_frequency_domain_features(segment, fs)
        hrv_morph_features = get_hrv_and_morphological_features(segment, fs)
        
        combined_features = np.concatenate([stat_features, freq_features, hrv_morph_features])
        all_feature_list.append(combined_features)
        
    return np.array(all_feature_list)