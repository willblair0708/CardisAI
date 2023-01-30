# CardisAI


The goal of this model is to accurately predict heart attacks based on ECG data from pacemakers. Apps, such as MyCareLink by Medtronic, enable pacemaker patients to view sensory data from their pacemakers in real time. These apps do not, however, feature functionality for patients to understand this health data.

## Summary 

Electrocardiography (ECG) is a key diagnostic tool to assess the cardiac condition of a patient. Automatic ECG interpretation algorithms as diagnosis support systems promise large reliefs for the medical personnel - only based on the number of ECGs that are routinely taken. However, the development of such algorithms requires large training datasets and clear benchmark procedures.

For instance, how might an average patient perceive an image of an ECG?

<div align="center">
<img src="https://user-images.githubusercontent.com/46399191/191921241-495090db-a088-46b6-bd09-0f7f21170b0a.png" height="350"/>
</div>

To most, this image has no functional importance. The use of AI, however, enables the potential for recognizing trends in ECG data from pacemakers to predict cardiac events, such as heart attacks, before they occur. This method is possible due to small changes in the Q-R-S movements in an ECG, which may indicate the development of plaque or atherosclerosis, and reveal negative trends in cardiac health. 

CardisAI is trained on a large selection of ECG data from pacemakers in order to provide clear and actionable insights to patients and their clinicians. 

In the future, I hope to incoroporate a combination of all sensory data from pacemakers (including accelerometers) to produce an improved pictrue of patient health. 

## Dataset

The [PTB-XL ECG dataset](https://physionet.org/content/ptb-xl/1.0.1/) is a large collection of 21837 clinical 12-lead ECGs, each 10 seconds in length, from 18885 patients. The raw waveform data has been labeled by up to two cardiologists, who added multiple ECG statements per record. The dataset includes 71 different ECG statements that adhere to the SCP-ECG standard and include diagnostic, form, and rhythm statements. This extensive annotation makes the dataset useful for training and testing algorithms for automatic ECG interpretation. Additionally, the dataset is augmented by metadata on demographics, infarction characteristics, likelihoods for diagnostic ECG statements, and annotated signal properties.

From the 18885 patients, 52% are male and 48% are female with ages covering the whole range from 0 to 95 years (median 62 and interquantile range of 22). The value of the dataset results from the comprehensive collection of many different co-occurring pathologies, but also from a large proportion of healthy control samples.

| Records | Superclass | Description |
|:---|:---|:---|
9528 | NORM | Normal ECG |
5486 | MI | Myocardial Infarction |
5250 | STTC | ST/T Change |
4907 | CD | Conduction Disturbance |
2655 | HYP | Hypertrophy |


The waveform files are stored in WaveForm DataBase (WFDB) format with 16-bit precision at a resolution of 1μV/LSB and a sampling frequency of 500Hz (records500/) beside downsampled versions of the waveform data at a sampling frequency of 100Hz (records100/).

All relevant metadata is stored in ptbxldatabase.csv with one row per record identified by ecgid and it contains 28 columns.

All information related to the used annotation scheme is stored in a dedicated scp_statements.csv that was enriched with mappings to other annotation standards.

# Requirements

This code was tested on Python 3 with Tensorflow 2.2. There is an older branch (tensorflow-v1) that contain the code implementation for Tensorflow 1.15.
You also need wfdb and opencv to process data from PTB-XL.
You can install the dependencies directly by creating a conda environment.

Install the dependencies (wfdb, pytorch, torchvision, cudatoolkit, fastai, fastprogress) by creating a conda environment:

    conda env create -f requirements.yml
    conda activate autoecg_env

### Get data
Download the dataset (PTB-XL) via the follwing bash-script:

    get_dataset.sh

This script first downloads [PTB-XL from PhysioNet](https://physionet.org/content/ptb-xl/) and stores it in `data/ptbxl/`.

If using Mac, ensure you make the following command to ensure your tensorflow is updated

``` 
pip install tensorflow-macos
```  

## Usage

    python main.py

This will perform all experiments for inception1d. 
Depending on the executing environment, this will take up to several hours. 
  

