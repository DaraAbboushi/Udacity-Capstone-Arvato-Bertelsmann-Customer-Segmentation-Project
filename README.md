# Udacity-Capstone-Arvato-Bertelsmann-Customer-Segmentation-Project

# Libraries:
1) Python 3
2) Seaborn
3) Matplotlib
4) Numpy
5) Pandas
6) SimpleImputer

# Description
This Capstone Project is the final part of Data Science Nanodegree Program by Udacity in collaboration with Arvato Bertelsmann.

The Project is divided into 2 main parts:

Customer Segmentation Report: In this section, the unsupervised learning technique such as PCA( principal component analysis) and clustering (k-means). Generally, These techniques are used to compare between the two provided datasets:  1) customers dataset. 2) The general population of Germany dataset.

Supervised Learning Model: Supervised Learning models are used to investigate mailout_train and use the model to predict on the mailout_test dataset which individuals are most likely to respond to a mailout campaign.


# Data files
azdias: demographics data for the general population of Germany;  891 211 persons (rows) x 366 features.

customers: demographics data for customers of a mail-order company; 191 652 persons (rows) x 369 features.

mailout_train: Demographics data for individuals who were targets of a marketing campaign; 42982 persons and 367 features including response of people.

mailout_test: Demographics data for individuals who were targets of a marketing campaign; 42833 persons and 366 features.

There are two more files which describes the attributes and its values. But the main datasets files are not available because of privacy of Arvato comapny's data.

Project Motivation
The main goal of this project is to characterize the customer segment of the population, and to build a model that will be able to predict customers for Arvato Financial Solutions.

# File Description
There are mainly two Notebooks available,

•  Customer Segmentation Part1.ipynb : It includes Data analysis and Unsurvised learning techinques to compare general population to the company's customers.

• AMachine Learning Models Part2.ipynb : It includes Supervised learning techniques to predict which individuals are most likely to respond to a mailout campaign.

And two python files,

• Customer Segmentation.py : It describes the data preprocessing and cleaning functions of azdias and customers dataset and unsupervised learning function.

• cleaning_functions.py : It describles the data preprocessing and cleaning functions of mailout_train and mailout_test dataset and model evaluation functions.

# Results
The main findings of the code can be found at this Customer Segemnetaion Report available: https://medium.com/@daraabboushi22/capstone-project-customer-segmentation-report-40579bcae541

Licensing, Authors, Acknowledgements
Udacity for providing such a Amazing project
Arvato Bertelsmann for providing datasets

References
1) https://medium.com/@zalarushirajsinh07/the-elbow-method-finding-the-optimal-number-of-clusters-d297f5aeb189
2) https://www.evidentlyai.com/classification-metrics/explain-roc-curve
3) https://builtin.com/data-science/step-step-explanation-principal-component-analysis
4) https://stackoverflow.com/questions/76703386/how-to-handle-imbalanced-data-in-a-classification-problem
5) https://developer.ibm.com/tutorials/awb-confusion-matrix-python/
