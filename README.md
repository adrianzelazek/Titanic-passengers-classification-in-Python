# Titanic passenger classification using Logistic Regression, Decision Tree, Random Forest and KNN in Python
### Author: Adrian Żelazek

In this project was used dataset concerning Titanic passengers. The dataset contains 891 observations as well as 12 variables. The target of the project is to build and select the best model that anticipates the death or survival of the Titanic passenger based on available passengers information using Logistic Regression, Decision Tree, Random Forest and KNN. 

The project includes Exploratory Data Analysis, grouping, visualization, splitting the training and testing data, Predictions as well as Model Evaluation. Furthermore, two models: Decision Tree and Random Forest have been modified by tuning of hyperparameters by GridSearchCV method. Models were evaluated by using indicators: Accuracy, Recall, Precision, F1, AUC. Moreover, Decision Tree after tuning of hyperparameters was also graphically evaluated by: Precision-recall curve, Accumulated profit curve, Lift curve, Forecasting error of classes and ROC curve. Furthermore, results of training and test datasets were compared with each other to exclude possible of overfitting. 

This project was developed for the purpose of practicing machine learning and data mining in Python. Furthermore, project was developed to train tuning of hyperparameters and visualization methods of classification model evaluation like: Precision-recall curve, Accumulated profit curve, Lift curve, Forecasting error of classes and ROC curve.
##### It is preferred to run/view the project via Jupyter Notebook (.ipynb), sometimes it is required to press "Reload" to load the file, than via a browser (HTML). To see the HTML version you first need to download the HTML file and then run it in your browser.

### Programming language and platform
* Python - version : 3.7.4
* Jupyter Notebook

### Libraries
* Scikit-learn - version: 0.22.2.post1
* Pandas - version: 0.25.1
* NumPy - version: 1.16.5
* Matplotlib - version: 3.1.2
* Seaborn - version: 0.9.0
* Missingno – version: 0.4.2
* Yellowbrick – version: 1.1

### Algorithms
* Logistic Regression
* Decision Tree
* Random Forest
* KNN
* GridSearchCV
* Pearson correlation coefficient
* Dummy Coding

### Methods of model evaluation
* Accuracy 
* Precision 
* Recall 
* F1 
* AUC 
* ROC curve
* Precision-recall curve 
* Accumulated profit curve
* Lift curve
* Forecasting error of classes

### Results
Decision tree after hyperparameter tuning presents definitely the best results. Random forest after hyperparameter tuning also presents high results which confirms that it is appropriate idea to tuning hyperparameters to get better results in model. The worst results were achieved by the KNN model. The best model (Decision Tree after tuning of hyperparameters) has: 
* percent of correct forecast = 86%,
* percent of positively classified positive results = 70%
* percent of correct positive forecasts = 91%
* harmonic average of precision and sensitivity = 0.79
* area under the roc curve = 0.92


