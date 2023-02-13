import matplotlib.pyplot as plt
import numpy as np
import heartpy as hp

sample_rate = 360


def load_visualise(data_file, annotations_file):
    
    #load the data
    ecg = hp.get_data(data_file)

    #get the annotations
    annotations = hp.get_data(annotations_file)
    #explore signal
    plt.figure(figsize=(12,3))
    plt.plot(ecg)
    plt.scatter(annotations, [ecg[int(x)] for x in annotations], color='green')
    plt.show()

    #and zoom in
    plt.figure(figsize=(12,3))
    plt.plot(ecg)
    plt.scatter(annotations, [ecg[int(x)] for x in annotations], color='green')
    plt.xlim(20000, 26000)
    plt.show()
    
    return ecg, annotations

ecg, annotations = load_visualise('ex1.csv', 'ex2.csv')


def filter_and_visualise(data, sample_rate):
    '''
    function that filters using remove_baseline_wander 
    and visualises result
    '''
    
    filtered = hp.remove_baseline_wander(data, sample_rate)

    plt.figure(figsize=(12,3))
    plt.title('signal with baseline wander removed')
    plt.plot(filtered)
    plt.show()

    plt.figure(figsize=(12,3))
    plt.title('zoomed in signal with baseline wander removed, original signal overlaid')
    plt.plot(hp.scale_data(data[200:1200]))
    plt.plot(hp.scale_data(filtered[200:1200]))
    plt.show()
    
    return filtered

filtered = filter_and_visualise(ecg, sample_rate)

def process_and_visualise(data, sample_rate):
    '''
    function that processes and visualises the data
    '''
    
    wd, m = hp.process(hp.scale_data(data), sample_rate)

    plt.figure(figsize=(12,4))
    hp.plotter(wd, m)

    for measure in m.keys():
        print('%s: %f' %(measure, m[measure]))
        
    #plot poincare
    hp.plot_poincare(wd, m)
    
    return wd, m

wd, m = hp.process(hp.scale_data(filtered), sample_rate)

plt.figure(figsize=(12,4))
hp.plotter(wd, m)

for measure in m.keys():
    print('%s: %f' %(measure, m[measure]))
        
#plot poincare
hp.plot_poincare(wd, m)

poincare_measures = ['sd1', 'sd2', 's', 'sd1/sd2']
for measure in poincare_measures:
    print('%s: %f' %(measure, m[measure]))

def resample_and_visualise(data, sample_rate):
    '''
    function that resamples the data and visualises the result
    '''
    
    from scipy.signal import resample

    resampled_signal = resample(data, len(data) * 4)

    wd, m = hp.process(hp.scale_data(resampled_signal), sample_rate * 4)

    plt.figure(figsize=(12,4))
    hp.plotter(wd, m)

    for measure in m.keys():
        print('%s: %f' %(measure, m[measure]))
        
    #plot poincare
    hp.plot_poincare(wd, m)
    
    return wd, m

plt.figure(figsize=(12,4))

hp.plotter(wd, m)

for measure in m.keys():
    print('%s: %f' %(measure, m[measure]))

ecg, annotations = load_visualise('118e12.csv', '118e12_ann.csv')

filtered = filter_and_visualise(ecg, sample_rate)

wd, m = hp.process(hp.scale_data(filtered), sample_rate)

plt.figure(figsize=(12,4))
hp.plotter(wd, m)

for measure in m.keys():
    print('%s: %f' %(measure, m[measure]))
        
#plot poincare
hp.plot_poincare(wd, m)


from scipy.signal import resample

resampled_signal = resample(filtered, len(filtered) * 4)

wd, m = hp.process(hp.scale_data(resampled_signal), sample_rate * 4)

plt.figure(figsize=(12,4))
hp.plotter(wd, m)

for measure in m.keys():
    print('%s: %f' %(measure, m[measure]))
        
#plot poincare
hp.plot_poincare(wd, m)

poincare_measures = ['sd1', 'sd2', 's', 'sd1/sd2']
for measure in poincare_measures:
    print('%s: %f' %(measure, m[measure]))


ecg, annotations = load_visualise('118e06.csv', '118e06_ann.csv')


filtered = hp.enhance_ecg_peaks(hp.scale_data(ecg), sample_rate, 
                                aggregation='median', iterations=5)

#show filtered signal
plt.figure(figsize=(12,4))
plt.plot(filtered)
plt.show()

#zoom in on signal section and overlay filtered segment 
plt.figure(figsize=(12,4))
plt.title('original signal zoom in')
plt.plot(hp.scale_data(ecg[15000:17000]), label='original data')
plt.title('processed signal zoom in')
plt.plot(hp.scale_data(filtered[15000:17000]), alpha=0.5, label='processed data')
plt.legend()
plt.show()


resampled_signal = resample(filtered, len(filtered) * 10)

wd, m = hp.process(hp.scale_data(resampled_signal), sample_rate * 10)

plt.figure(figsize=(12,4))
hp.plotter(wd, m)

for measure in m.keys():
    print('%s: %f' %(measure, m[measure]))
        
#plot poincare
hp.plot_poincare(wd, m)

poincare_measures = ['sd1', 'sd2', 's', 'sd1/sd2']
for measure in poincare_measures:
    print('%s: %f' %(measure, m[measure]))


ecg, annotations = load_visualise('118e00.csv', '118e00_ann.csv')



filtered = hp.enhance_ecg_peaks(hp.scale_data(ecg), sample_rate, 
                                aggregation='median', iterations=4)

plt.figure(figsize=(12,4))
plt.plot(filtered)
plt.show()

plt.figure(figsize=(12,4))
#plt.subplot(211)
plt.title('original signal zoom in')
plt.plot(hp.scale_data(ecg[15000:17000]), label='original data')
#plt.subplot(212)
plt.title('processed signal zoom in')
plt.plot(hp.scale_data(filtered[15000:17000]), alpha=0.5, label='processed data')
plt.legend()
plt.show()

resampled_signal = resample(filtered, len(filtered) * 10)

wd, m = hp.process(hp.scale_data(resampled_signal), sample_rate * 10)

plt.figure(figsize=(12,4))
hp.plotter(wd, m)

for measure in m.keys():
    print('%s: %f' %(measure, m[measure]))
        
#plot poincare
hp.plot_poincare(wd, m)

poincare_measures = ['sd1', 'sd2', 's', 'sd1/sd2']
for measure in poincare_measures:
    print('%s: %f' %(measure, m[measure]))


filtered = hp.filter_signal(ecg[0:14500], 0.05, sample_rate, filtertype='notch')

wd, m = hp.process(hp.scale_data(filtered), sample_rate)

plt.figure(figsize=(12,4))
hp.plotter(wd, m)

for measure in m.keys():
    print('%s: %f' %(measure, m[measure]))
        
#plot poincare
hp.plot_poincare(wd, m)

poincare_measures = ['sd1', 'sd2', 's', 'sd1/sd2']
for measure in poincare_measures:
    print('%s: %f' %(measure, m[measure]))




