import numpy as np
import scipy.signal
import matplotlib.pyplot as plt

def detect_peaks(ecg_signal, threshold=0.5):
    # Filter the ECG signal
    low_pass = 5.0
    b, a = scipy.signal.butter(5, low_pass, 'lowpass')
    filtered_ecg = scipy.signal.filtfilt(b, a, ecg_signal)
    
    # Find the R-peaks in the ECG signal
    r_peaks, _ = scipy.signal.find_peaks(filtered_ecg, height=threshold)
    
    return r_peaks

def analyze_ecg_signal(ecg_signal, sampling_rate=500):
    # Calculate the heart rate
    r_peaks = detect_peaks(ecg_signal)
    heart_rate = len(r_peaks) * (sampling_rate / len(ecg_signal))
    
    return heart_rate, r_peaks

def process_ecg_signal(ecg_signal):
    # Filter the ECG signal
    low_pass = 5.0
    b, a = scipy.signal.butter(5, low_pass, 'lowpass')
    filtered_ecg = scipy.signal.filtfilt(b, a, ecg_signal)
    
    return filtered_ecg

def detect_features(psd, f, threshold=0.5):
    # Normalize the power spectral density
    psd = psd / np.sum(psd)
    
    # Find the peaks in the power spectral density
    peaks, _ = scipy.signal.find_peaks(psd, height=threshold)
    
    return peaks, f[peaks]

def ecg_power_spectral_density(ecg_signal, sampling_rate=500):
    # Calculate the power spectral density of the ECG signal
    f, psd = scipy.signal.welch(ecg_signal, fs=sampling_rate, nperseg=256)
    
    return f, psd

def heart_rate_variability(r_peaks, sampling_rate=500):
    # Calculate the heart rate variability
    r_peaks = np.array(r_peaks)
    rr_intervals = np.diff(r_peaks) / sampling_rate
    rmssd = np.sqrt(np.mean(np.square(np.diff(rr_intervals))))
    
    return rmssd

def cardiac_output(heart_rate, rmssd):
    # Calculate the cardiac output
    cardiac_output = 60 * heart_rate / rmssd
    
    return cardiac_output

# Load the ECG data
ecg_data = np.loadtxt('ecg_signal.txt')

# Analyze the ECG data
heart_rate, r_peaks = analyze_ecg_signal(ecg_data)

# Plot the ECG signal and the detected R-peaks
plt.plot(ecg_data)
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.title('ECG Signal')
plt.plot(r_peaks, ecg_data[r_peaks], 'ro')
plt.show()

# Print the heart rate
print('Heart Rate:', heart_rate, 'bpm')
