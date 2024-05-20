# MidLog
This is the basic implementation of MidLog: An Automated Log Anomaly Detection Method based on Multi-head GRU 

# Datasets
We used 2 common open-source log datasets, HDFS and BGL. Their detailed information is as follows.

| Data | Description | Details | DownLoad Link |
| :---: | :---: | :---: | :---: |
| HDFS | Hadoop distributed file system log |[LogHub](https://github.com/logpai/loghub/tree/master/HDFS)| [HDFS(Organized by Loghub)](https://zenodo.org/records/8196385)|
| BGL | Blue Gene/L supercomputer log |[THE HPC4 DATA](https://www.usenix.org/cfdr-data#hpc4)| [BGL(Organized by Loghub)](https://zenodo.org/records/8196385) |

# Preparation

## Key package
pytorch 3.8<br>
python 1.7.0<br>
tqdm<br>

## Log parsing
We used an existing log parsing method based on regular expressions, i.e., [logdeep](https://github.com/donglee-afar/logdeep.git). This repository provides the parsing method for two datasets and partial parsed data.

# Anomaly detection
 Go to demos/multi_RNN_demo.py and set necessary parameters(or use default parameters). Set mode to 'train' to train MidLog
