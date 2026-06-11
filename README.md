# Support Material for Objective Function Impact on Signatures

This repository contains MATLAB and Python code used for running, analyzing and plotting a comparative analysis on the impact of objective function choice on hydrologic signatures.

## Structure

The repository is separated into three major components:

- **`00_hpc_workflows`**: Scripts used to compile and calibrate the MATLAB runs on the HPC (thanks to the TU Dresden – Zentrum für Informationsdienste und Hochleistungsrechnen).
- **`01_preparation`**: Python scripts run prior to the analysis, including the benchmarking routine and catchment selection through clustering.
- **`02_analysis_plotting`**: Scripts necessary to reproduce the figures shown in the paper.
  - `02a_combine_hpc_runs`: Needs to be run first and requires the data from the repository linked below. Extraction of 500 runs from the 18,800 combinations takes roughly 10 minutes on an M3 MacBook Pro. Additionally, there is a MATLAB script extracting the benchmark scores from the Python script.
  - `02b_analysis`: MATLAB scripts for generating the plots used in the paper.

> These scripts were only tested on the specific machines they were run on. Do not hesitate to reach out if you have issues running the code.

## Data

The data used in the analysis, including the performance and signature values and an index of the runs (~30 GB), is available on HuggingFace:

👉 https://huggingface.co/datasets/pewag/data_signature_objective_function

Download the files and place them into:

```
data/
```

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
```
