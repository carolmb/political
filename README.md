### A complex network approach to political analysis: application tothe Brazilian Chamber of Deputies

[Here](https://arxiv.org/pdf/1909.02346.pdf) to read the paper.

#### Software dependencies

This project is built using Python 3. It is necessary pip3 to install the dependencies.
	
> pip3 install -r requirements.txt

#### Run the project: 

This script will preprocess the original data, generate networks and calculate some metrics mentioned in the paper (fragmentation/isolation).  

> chmod +x run.sh
> ./run.sh

To generate plots of metrics (degree, modularity, diversity):

> python3 data_metrics.py