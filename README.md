# CardisAI


The goal of this model is to accurately predict heart attacks based on ECG data from pacemakers. Apps, such as MyCareLink by Medtronic, enable pacemaker patients to view sensory data from their pacemakers in real time. These apps do not, however, feature functionality for patients to understand this health data.

For instance, how might an average patient perceive this image?


To most, this image has no functional importance. The use of AI, however, enables the potential for recognizing trends in ECG data from pacemakers to predict cardiac events, such as heart attacks, before they occur. This method is possible due to small changes in the Q-R-S movements in an ECG, which may indicate the development of plaque or atherosclerosis, and reveal negative trends in cardiac health. 

CardisAI is trained on a large selection of ECG data from pacemakers in order to provide clear and actionable insights to patients and their clinicians. 

In the future, I hope to incoroporate a combination of all sensory data from pacemakers (including accelerometers) to produce an improved pictrue of patient health. 


# Requirements

This code was tested on Python 3 with Tensorflow 2.2. There is an older branch (tensorflow-v1) that contain the code implementation for Tensorflow 1.15.
You also need wfdb and opencv to process data from PTB-XL.
You can install the dependencies directly by creating a conda environment.

``` 
conda env create -f ecg_env.yml
conda activate ecg_env
```  
If using Mac, ensure you make the following command to ensure your tensorflow is updated

``` 
pip install tensorflow-macos
```  
