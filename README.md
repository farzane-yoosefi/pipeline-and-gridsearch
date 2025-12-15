>> These notes serve to demonstrate my machine learning knowledge to potential employers while also helping beginners learn complex concepts in simple terms.

## Gridsearch and Pipeline
Before reading this page you need to know about pipeline and gridsearch .
If you don't know about each of them please refer to these pages :
`Grid search :` https://github.com/farzane-yoosefi/Gridsearch  
`pipeline :` https://github.com/farzane-yoosefi/pipleline

### GridsearchCV
As you know `GridsearchCV` is a machine learning tool that uses cross-validation to systematically search through a set of hyperparameters you provide to find the best combinations that gives the best performance.

It finds the best combination from the list you provide in parameter grid .It can't find them from outside of the list.

Think of it as a powerfull,systematic search within a predefined list of values- not magical discovery.

### Pipleline
pipeline uses defined steps to control the workflow from raw data to trained model ,
ensuring efficiency and consistent process in both train and test data set.

>> `Pipeline` ensures preprocessing is correctly fitted on the training data during cross-validation. `GridSearchCV` searches the hyperparameters of the estimator and also the parameters of each step within the pipeline.Together, they prevent data **leakage** and create **optimized** models.

## Implementation
In this code we are going to train a SVS model on iris dataset.
The model is placed inside the pipeline in the final step as estimator.
### Step1 :
First import the necessary libraries
```python
rom sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import classification_report
```
### Step2 :
Import the model and define feature and target
```python
from sklearn.datasets import load_iris
data = load_iris()
X = data.data
Y= data.target
```
split the data
```python
X_train , X_test , Y_train,Y_test = train_test_split(X , Y , test_size= 0.2, random_state=34)
```
### Step 3 : 
Our data is very simple and needs less preprocessing; therefore, pipeline contains few steps.

Now Create your Pipeline :
```python
pipe = Pipeline([
    ('scaler' , StandardScaler()),
    ('classifier', SVC())
])
```
### step 4 :
We want to define the parameters for the gridsearch but what if we don't know what parameters look like? so in order to see them use this code :
`pipe.get_params().keys()`
```python
pipe.get_params().keys()
```
With this code all possible parameters area availble.
Output :
```python
dict_keys(['memory', 'steps', 'transform_input', 'verbose', 'scaler', 'classifier', 'scaler__copy', 'scaler__with_mean', 'scaler__with_std', 'classifier__C', 'classifier__break_ties', 'classifier__cache_size', 'classifier__class_weight', 'classifier__coef0', 'classifier__decision_function_shape', 'classifier__degree', 'classifier__gamma', 'classifier__kernel', 'classifier__max_iter', 'classifier__probability', 'classifier__random_state', 'classifier__shrinking', 'classifier__tol', 'classifier__verbose'])
```
Define the parameters :
```pyton
param_grid ={
    'scaler__with_mean': [True, False],
    'classifier__C': [0.1, 1, 10], 
    'classifier__kernel': ['linear', 'rbf', 'poly'],
    'classifier__gamma': ['scale', 'auto', 0.1, 1]
}
```
### Step 5 :
Now the code is ready for ceating a `GridsearchCV`.
- pipeline plays the **estimator** rule
- **cv** is the number of foldd in cross-validation
- **scoring='accuracy'**:Evaluate each model using accuracy
- **n_jobs** use all available CPU cores
     - **n_jobs = 1** runs slower
     - **n_jobs = 2** Uses 2 CPU cores in parallal
     - **n_jobs = -1** Use all CPU cores (fastest)
     - **verbose = 1** 
- **verbose** makes gridsearch show a progress bar and massages so you can see what is happening.
     - 0 = Silent (no output)
     - 1 = Moderate (progress + completion)
     - 2 = Detailed (per-fold results)
     - 3 = Very detailed
     - 4+ = Maximum verbosity (everything)
 - **return_train_score=True** It tells GridsearchCV to calculate and save the training score for each parameter combination.
```python
grisdearch = GridSearchCV( 
    estimator=pipe,
    param_grid=param_grid,
    cv = 5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1,
    return_train_score=True
)
```
### step 6 :
##### train the model :
```python
grisdearch.fit(X_train,Y_train)
```
Output :
```
Fitting 5 folds for each of 72 candidates, totalling 360 fits
```
Can you tell me where is this result from ?
Answer is : **verbose**
Also this is the cross-validation result coming from gridsearch,just showing the GridSearchCV configuration.
<p align="center"><img src="https://github.com/farzane-yoosefi/pipeline-and-gridsearch/blob/main/SVS.JPG" alt="cross-validatiom table" width="300" /></p>

#### Do prediction 
```python
y_pred = grisdearch.predict(X_test)
```
## Step 7 :
Verify your model :
```python
score = classification_report(y_pred,Y_test)
print(score)
```
Output :
```
precision    recall  f1-score   support

           0       1.00      1.00      1.00        11
           1       1.00      0.94      0.97        16
           2       0.75      1.00      0.86         3

    accuracy                           0.97        30
   macro avg       0.92      0.98      0.94        30
weighted avg       0.97      0.97      0.97        30
```
Interpret the output :  


| Class | Precision → Meaning | Recall → Meaning | F1 → Meaning | Support |
|-------|---------------------|------------------|--------------|---------|
| **Overall** | - → - | - → - | - → - | 30 → **97% accuracy** |
| **0** | 1.00 → Every prediction correct | 1.00 → Found all 11 samples | 1.00 → Perfect balance | 11 samples |
| **1** | 1.00 → All "class 1" predictions correct | 0.94 → Found 15/16 samples | 0.86 → Excellent balance | 16 samples |
| **2** | 0.75 → Correct 3/4 times | 1.00 → Found all 3 samples | 0.86 → Good but imperfect | 3 samples |


