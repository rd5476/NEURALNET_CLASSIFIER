# NEURALNET_CLASSIFIER
Readme:


Author :Rahul Dashora

Inputs:
train_data.csv
test_data.csv

Execution files:
trainMLP
executeMLP

Steps:

1. MLP

	1.Run the trainMLP.py file
	2.Enter train_data.csv on the terminal when asked'Enter the filename'
	3.The terminal displays two 2 plots
 		a) Plot of SSE vs Epoch 
  		b) Plot of the training data set.
		c) 5 weight files are generated through this
	4.Now run the executeMLP.py file 
	5.Enter test_data.csv in the command line when asked 'Enter the file name'.
		(Weight files must be in the same directory)
	6.The terminal displays
		a)Confusion Matrix 
		b)Recognition Rate 
		c)Profit
		d)Plot for test sample classification region for different epochs.

