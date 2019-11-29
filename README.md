### A complex network approach to political analysis: application to the Brazilian Chamber of Deputies

[Here](https://arxiv.org/pdf/1909.02346.pdf) to read the paper.

### Software dependencies

This project is built using Python 3. It is necessary pip3 to install the dependencies.
	
> pip3 install -r requirements.txt

### Running

This script will preprocess the original data, generate networks and calculate some metrics mentioned in the paper (fragmentation/isolation).  

> chmod +x run.sh

> ./run.sh

To generate plots of metrics (degree, modularity, diversity):

> python3 data_metrics.py

### MIT License

Copyright (c) 2019 carolmb

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
