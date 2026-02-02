import pandas as pd
import numpy as np
import os
import wfdb
from scipy.signal import butter, filtfilt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

def clean_metadata(df):
    """
    Metadata'daki aykırı ve eksik değerleri temizler.
    """
    median_age = df['age'].median()
    df['age'] = df['age'].fillna(median_age)

    mean_height = df['height'].mean()
    mean_weight = df['weight'].mean()
    df['height'] = df['height'].fillna(mean_height)
    df['weight'] = df['weight'].fillna(mean_weight)
    
    if df['sex'].isnull().any():
        mode_sex = df['sex'].mode()[0]
        df['sex'] = df['sex'].fillna(mode_sex)

    return df

def encode_categorical_features(df):
    """
    Kategorik özellikleri (hedef ve cinsiyet) sayısal formata dönüştürür.
    """
    le = LabelEncoder()
    df['main_diagnostic_encoded'] = le.fit_transform(df['main_diagnostic'])
    
    ohe = OneHotEncoder(sparse_output=False, drop='if_binary') 
    sex_encoded = ohe.fit_transform(df[['sex']])
    df['sex_encoded'] = sex_encoded.flatten().astype(int)
    
    return df, le

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    """
    Bant geçiren filtre uygular.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y

def preprocess_signals(df, data_path, sampling_rate=100):
    """
    Tüm EKG sinyallerini yükler ve gürültü filtresi uygular.
    """
    signals = []
    
    all_filepaths = df['filepath'].unique()

    for filepath in all_filepaths:
            signal, meta = wfdb.rdsamp(filepath.replace('.dat', ''))
            
            filtered_signal = np.array([butter_bandpass_filter(signal[:, i], 0.5, 40, sampling_rate) for i in range(signal.shape[1])])
            
            signals.append(filtered_signal.T)
            
    df['filtered_signal'] = signals
    df.dropna(subset=['filtered_signal'], inplace=True)
    
    return df

def create_segments(df, segment_length=250, overlap=150):
    """
    Filtrelenmiş sinyalleri segmentlere böler.
    250 data point = 2.5 saniye (100Hz için)
    150 data point overlap = 1.5 saniye (%60 overlap)
    """
    X, y, groups = [], [], []
    
    for index, row in df.iterrows():
        signal = row['filtered_signal']
        label = row['main_diagnostic_encoded']
        
        for i in range(0, len(signal) - segment_length + 1, segment_length - overlap):
            segment = signal[i:i+segment_length]
            X.append(segment)
            y.append(label)
            groups.append(index)
            
    return np.array(X), np.array(y), np.array(groups)


def run_preprocessing_pipeline(df, data_path):
    """
    Tüm ön işleme adımlarını sırayla çalıştıran ana fonksiyon.
    """
    df = clean_metadata(df)
    df, label_encoder = encode_categorical_features(df)
    
    df = preprocess_signals(df, data_path)
    
    X_segments, y_segments, patient_groups = create_segments(df, segment_length=250, overlap=150)
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_segments, y_segments, test_size=0.2, random_state=42, stratify=y_segments
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), label_encoder