# DAProject

In order to run the 4 different models' experiments the following steps need to be followed.

Step 1: Download the tenrec dataset from https://static.qblv.qq.com/qblv/h5/algo-frontend/tenrec_dataset.html and extract the sbr_data_1M.csv, cold_data.csv, cold_data_0.3.csv, cold_data_0.7.csv, cold_data_1.csv  into the folder named Data. If the folder does not exist, create one.

Step 2: Run the following commands for the different experiments:

For Bert4Rec:
```python
python main.py --model='bert'
```
For Peter4Rec:
```python
python main.py --model='peter4rec'
```
For RNN:
```python
python main.py --model='rnn'
```
For KNN:
```python
python main.py --model='knn'
```

To be noted the logs of odds and odds ratio were not used as the main bias metric, as the models results made it not possible to calculate.
