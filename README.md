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

# Data Lake Access Guide
This document provides instructions on how to access and use the data lake infrastructure developed by IFAB team.

## Data Lake structure
![Datalake structure composed by: 2 routers, a Virtualk Machine (VM) for the Data Lake (orange) where is MongoDB, 6 VM for the compute nodes (green) among which the last one is the Login Node where is attached a volume with MinIO installed.](images/Datalake_structure.png)

The Data Lake is composed by 7 Virtual Machines (VM) connected to the public network via 2 routers that controll the private networks and organize the comunication between the VMs. One router is connected to the Data Lake VM (orange), in this VM is installed Mongo DB which allows to have a Data Catalogue where all the file stored have specific metadata allowing for queries. The other 6 VMs (green) represent 5 Computing Nodes and 1 (the last in the graph) Login Node. 
The Login Node is the most important one since has multiple crucial roles in the usage of the Data Lake:
- **File storage**: to the login node is attached a Volume where is installed MinIO.  
- **Roles**: the division of the Login Node in roles allows for private and shared folders.
- **Job Launch**: via the Login Node is possible to launch jobs, see the job queue. (link to the section ffor job launching)

To upload, download fiels or make a query its necessary to interface with the DataLake, further inforamations are reported in the WP2 > dl_client > [README](https://github.com/ifabfoundation/AIND/blob/main/WP-2/dl_client/README.md)

### MongoDB
MongoDB is a Data Catalogue which stores the metadata uploaded with the file. It is attached to the Data Lake Virtual Machine.

### MinIO
MinIO is a software that allows to mirror the volume storage, which is tipically folder based, as an object storage wich is more efficent for operations with the data.
Object storage is based on buckets and then in each buket the files are organised by prefix (which to us looks like a folder but tecnically has different properties)
This software is installed to the volume attached to the login node. 

It is possible to access MinIO via terminal or via [website](http://131.175.204.159:9001/). Via terminal you can list the files in the Volume after connecting with the [Login Node](https://github.com/ifabfoundation/AIND/tree/ChiaraPifab-patch-1?tab=readme-ov-file#2-connect-to-the-loghin-node) by:
```bash
# access the data folder in the login node rooth
cd /data
# enter the bucket
cd aind
# list file and folders
ls
```

However, it is not usefull to access MinIO since we can only view the content of the buckets, see file names and the queries folders. Even if possible files should not be directly uploaded to MinIO because this process would not include MongoDB and thus we would not have metadata linked to the file and consequently no query can be performed on that file.

### Roles
The login node has roles eachone with specific privileges and permissions. Each role has its private folder (IFAB, POLIBA, UNINA), furthere there is a common folder (SHARED) where is possible to share files. The admin role (aind) has access to all the folders. 

In the login node folder you can upload codes and launch jobs, but you **must NOT** upload here data, no real data, no generated data and no resutls. This storage is not certified for clinical data (the datalake VM is the only ISO27001 certified infrastructure for data storage & manipolation) further file uploaded to the folders do not have metadata and queries are not supported.

On the machine is not possible to upload python libraries, for this reason it is necessary to create a virtual environment in each folders where to import all libraries tha might be needed. In the shared folder is already present a Virtual environment with the most common libraries and dl_client, this library is extensively explained **here (link)**, which allows to interact via terminal with the Data Lake. It is necessary to install the dl_client library in every virtual environment to be able to interact with the Data Lake.

## Prerequisites

- Ubuntu operating system or [WSL](https://ubuntu.com/desktop/wsl)
- Basic command line knowledge

## Access to the Login Node
It is possible to access the login node via SSH using the Ubuntu terminal. The acces is allowed through a "personal" key that gives you access to your role. There is a key-role for each institution in the project.

Accessing to the login node and role-folders is necessary to launch jobs in the computing nodes, see the jobs queue, interface with the datalake via command line.

Now lets see how to activate the SSH key, update the environment, access to the login node and create an alias for easier access.

### 1. SSH Key Installation
You will receive a key_name.pem SSH key file, the name of the key is based on your institution and the possible key_names are: IFAB.pem, POLIBA.pem, UNINA.pem. In this guide we will refere to it as key_name.pem.

This key must be installed in your Ubuntu SSH directory.

1. Download the key_name.pem and, if you are using Windows, move the key from the downlods folder to the *C:/Users* folder, in this way it will be easier to find it.
2. To move the key_name.pem to the SSH directory in Ubunto its necessary to operate from the Ubuntu terminal:
```bash
mv <key_path>/key_name.pem ~/.ssh/  

mv /mnt/c/Users/key_name.pem ~/.ssh/   (if the key is the User Windows folder)
```
3. Set the correct permissions for the key file: 
```bash
chmod 600 ~/.ssh/key_name.pem
```

### 2. Connect to the Loghin node
Connect to the login node of the HPC environment through SSH command (<ROLE> is your institute, so sostitute it with the appropriate one among POLIBA / UNINA / IFAB):
```bash
ssh -i ~/.ssh/key_name.pem <ROLE>@131.175.204.159
```

### 3. Create an alias
For qucker acces to the loghin node and not have to type the last command lines every time you can create an allias that will allow you to acces to the login node with only a comand.

1. From the Ubuntu terminal (wsl) creat/open the file .bash_aliases
```bash
nano ~/.bash_aliases
```
2. Inside the file wirite a new alias
```bash
alias <NAMEssh>='ssh -i ~/.ssh/key_name.pem <ROLE>@131.175.204.159'
```
The alias that you have chosen (<NAMEssh>) will be the command line that you can use to directly acces the login node.
3. Exit the file --> 'Ctrl X' and then 'Enter'

4. Update your environment:
```bash
source ~/.bashrc
```

### 4. Navigation
After accessing the login node you'll find yourself directly into your Role folder, where by typing `ls` ypu can see the content of the folder.

The command `cd` allwos to move in side the folders and `cd ..` goes backword. 

#### Access the SHARED folder
The traditional way would be to reach the *home* folder and then move into the *SHARED* folder.
If you are in your role folder the commands are:
```bash
cd ..
cd SHARED
```

However there is a special command implemented that allows you to reache the *SHARED* folder any time from any place in the loghin node:
```bash
shared
```

### Virtual Environments
It is not possible to install libraries and packages in the machine. Thus Virtual Environments (*venv*) are needed.

#### ⟶ Activate the default Virtual Envirmnemt
You can find a `default` virtual environment in the `SHARED/ml_environments` folder.
This environment can be activate any time you are in the login node, following the general comand line:
```bash
source default/bin/activate      (using pyvenv, Ubuntu)

conda activate default           (using conda, Windows terminal)    ##########NOT SURE IT IS NEEDED HERE
```
Another option, specific for this environment, is using the custom comand:
```bash
defaultenv
```

#### ⟶ Create a new Virtual Environment
Where do I want to save this environment?????
If you are using conda
```bash
conda create <venv_name>

```

if you are using pyvenv
```bash
python3 -m venv <venv_name>

```

#### ⟶ Activate a Virtual Environment
Where do I want to save this environment?????

If you are using conda
```bash
conda activate <venv_name>

```

if you are using pyvenv
```bash
source <venv_name>/bin/activate

```

#### ⟶ Deactivate a Virtual Environment
To deactivate the Virtual Environment you have activated and worked in, you can use the comand:
```bash
deactivate
```

### Job launch from the Login Node
... (bash commands)

## Interaction with the Data Lake
To interface with Data Lake, and thus be able to upload, download, and search for files, there are two options:
- via command line from terminal (https://github.com/ifabfoundation/AIND/blob/main/WP-2/dl_client/README.md#command-line-interface)
- using the dl_client library (https://github.com/ifabfoundation/AIND/tree/main/WP-2/dl_client#dl-client) from vscode ----> ***OR WHEN????***

The main difference between the command line and the dl_client library is that in the former case you act locally so you upload and download files to your computer, which must not be done with medical data due to security and privacy rules.
Using the dl_client library its possible to use vscode and other platforms allowig to open the data in the proper format (pd.DataFrame, ...), view and create modified versions of the data without downloading the file locally. In this case it is necessary to have the library in the environment we are working in and in the local folder where we have the codes we are running on vscode.

All that is needed to know is in the README (https://github.com/ifabfoundation/AIND/tree/main/WP-2/dl_client/README.md).

## Important commands
List the file in a directory ⟶    `ls` 
Move in the directory ⟶    `cd` `cd ..`
Access the *SHARED* folder ⟶    `shared`
Activate the default environment ⟶    `defaultenv`
Deactivate a virtual envirnment ⟶    `deactivate`
Check the job queue ⟶     `squeue`
Exit from login node ⟶    `exit`
## Support
If you encounter any issues or have questions regarding access or usage, please contact us:
raimondo.reggio@ifabfoundation.org
benedetta.baldini@ifabfoundation.org
chiara.pollicini@ifabfoundation.org

