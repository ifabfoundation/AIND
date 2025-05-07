# AIND

Personalized Data-Driven Prevention of Neurodegenerative Disorders: A Datalake & Artificial Intelligence Approach

## Repository Description

This repository contains the code scripts for the AIND project, organized across different workpages. It is divided into three main folders, each corresponding to a specific workpage:

- **WP2**: Datalake Building
- **WP3**: Neurodegenerative Disease Modelling
- **WP4**: Predictive A.I. Algorithms

## Project Goals

### Primary Objective

Develop artificial intelligence personalized solutions as ‘proof-of-concepts’ to determine the individual risk in adult people of contracting neurodegenerative disorders (NDD), i.e., Parkinson’s or Alzheimer’s diseases, based on data derived from a multiscale health NDD Datalake built from available clinical databases, available disease models, and state-of-the-art predictive algorithms.

### Secondary Objectives

- **Data Accessibility**: Identify and make accessible for databasing reliable data and information regarding patients and scientific publications on Parkinson's Disease (PD) and/or Alzheimer's Disease (AD) in their early stages of disease progression.

- **Automatic Data Extraction**: Provide an initial ‘proof-of-concept’ for establishing an automatic data extraction process to obtain information from key clinical and scientific databases and keep the Datalake constantly updated.

- **Predictive Models for Precision Medicine**: Identify and operationalize the most up-to-date predictive/prognostic disease models to provide data to drive an initial training set for the precision medicine AI algorithms, as well as real data-driven comparators for validation.

## POLIBA Data Lake Access Guide
This document provides instructions on how to access and use the data lake infrastructure developed by our team.

### Prerequisites

- Ubuntu operating system
- SSH client
- Basic command line knowledge

### Initial Setup
### 1. SSH Key Installation
You will receive a POLIBA.pem SSH key file. This key must be installed in your Ubuntu SSH directory:

1. Download the POLIBA.pem key file provided to you
2. Move the key to your SSH directory:
```
mv POLIBA.pem ~/.ssh/
```
3. Set the correct permissions for the key file:
```
chmod 600 ~/.ssh/POLIBA.pem
```
### 2. Environment Update
After installing the key, update your environment:
```
source ~/.bashrc
```
### 3. Role Activation
Connect to the login node of the HPC environment through SSH command:
```
cd ~/.ssh
ssh -i ~/.ssh/POLIBA.pem POLIBA@131.175.204.159
```

### 4. Navigation
Once connected, you can navigate to the home directory:
```
cd ..
```
From this directory, you will have access to the data lake resources.

### Usage
....
   
### Support
If you encounter any issues or have questions regarding access or usage, please contact us:
raimondo.reggio@ifabfoundation.org
benedetta.baldini@ifabfoundation.org
chiara.pollicini@ifabfoundation.org

