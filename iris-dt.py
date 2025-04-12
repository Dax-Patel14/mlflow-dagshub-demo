import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend before importing pyplot
from matplotlib import pyplot as plt
import seaborn as sns
import os

import dagshub
dagshub.init(repo_owner='Dax-Patel14', repo_name='mlflow-dagshub-demo', mlflow=True)


mlflow.set_tracking_uri("https://dagshub.com/Dax-Patel14/mlflow-dagshub-demo.mlflow")

# Load iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# split into train and test
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# Define params for Random forest
max_depth = 2



# Apply MLFLOW
#this run should be considered in iris-dt exp
mlflow.set_experiment('iris-dt') # if it is not avilable in mlflow then it will be created

with mlflow.start_run() as run: # COntext manager

    dt = DecisionTreeClassifier(max_depth=max_depth)
    dt.fit(X_train,y_train)

    y_pred = dt.predict(X_test)

    accuracy = accuracy_score(y_test,y_pred)

    mlflow.log_metric('accuracy',accuracy)
    mlflow.log_param('max_depth',max_depth)
   
    # create confusion matrix plot
    cm = confusion_matrix(y_test,y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',xticklabels=iris.target_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('COnfusion matrix')

    # save the plot as an artifacts
    plt.savefig("Confusion_matrix.png")

    #mlflow plot log
    mlflow.log_artifact("Confusion_matrix.png")

    #mlflow code log
    mlflow.log_artifact(__file__)

    #mlflow model log
    mlflow.sklearn.log_model(dt,"decision tree")

    #mlflow tags log
    mlflow.set_tag('author','Daksh')
    mlflow.set_tag('model','decision tree')

    print('accuracy',accuracy)

