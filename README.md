# Hydrologic Signatures Analysis

This repository contains MATLAB code used for generating hydrologic signatures and model evaluation for my thesis.

## Data

Large data files (`.mat` and `.nc`) are not stored in the repository.

Download them here:
https://drive.google.com/drive/folders/1ABWhnASsVPEbazySIYHubWO9ghRXlFd0?usp=drive_link

Place all files into:
`data/`

## Dependencies

This project uses two external toolboxes:

- **TOSSH** – https://github.com/TOSSHtoolbox/TOSSH  
- **MARRMoT** – https://github.com/wknoben/MARRMoT

Download them and place them anywhere on your computer.

## Setup

Add the toolboxes to your MATLAB path:

```matlab
addpath(genpath('/path/to/TOSSH'));
addpath(genpath('/path/to/MARRMoT'));
