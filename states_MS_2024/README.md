## Energetically costly functional network dynamics in cognitively impaired multiple sclerosis patients
This repo contains all files associated with this manuscript by Broeders et al., which is available as a preprint at xxxx

### State Dynamics based on Edge Time Series
`EdgeTS_Kmeans_States.m`

This code performs k-means clustering on edge time series data to deteremine state dynamics.

- Prerequisites: The fMRI data needs to be pre-processed
- Input: Pre-processed fMRI data and a file indicating which regions to include (to account for distortion)
- Output: State sequence and transition parameters

### Control Energy
`Control_Energy.py`

Calculate the minimum control energy associated with state-transitions

 - Prerequisites: The states need to have been defined
 - Input: The fMRI time series, DTI connectivity matrices, a state sequence, and the included regions
 - Output: The transition control energy values
