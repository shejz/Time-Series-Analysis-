# -*- coding: utf-8 -*-

## How to Create & Transpose a Vector or Matrix
def Kickstarter_Example_1():
    print()
    print(format('How to Create/Transpose a Vector and/or Matrix', '*^75'))
    
    # Load library
    import numpy as np
    
    # Create vector
    vector = np.array([1, 2, 3, 4, 5, 6])
    print()
    print("Original Vector: \n", vector)
    # Tranpose vector
    V = vector.T
    print("Transpose Vector: \n", V)
    
    # Create matrix
    matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])
    print()
    print("Original Matrix: \n", matrix)    
    # Transpose matrix
    M = matrix.T    
    print("Transpose Matrix: \n", M)
Kickstarter_Example_1()

## How to Create A Sparse Matrix
def Kickstarter_Example_2():
    print()
    print(format('How to Create A Sparse Matrix', '*^50'))
    
    # Load libraries
    import numpy as np
    from scipy import sparse
    
    # Create a matrix
    matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])
    print()
    print("Original Matrix: \n", matrix) 

    # Create sparse matrices
    print()
    print("Sparse Matrices: ")     
    print()
    print(sparse.csr_matrix(matrix))
    print()
    print(sparse.bsr_matrix(matrix))
    print()
    print(sparse.coo_matrix(matrix))
    print()
    print(sparse.csc_matrix(matrix))
    print()
    print(sparse.dia_matrix(matrix))
    print()
    print(sparse.dok_matrix(matrix))
    print()
    print(sparse.lil_matrix(matrix))
    print()
Kickstarter_Example_2()

## How to Select Elements from a Numpy Array
def Kickstarter_Example_3():
    print()
    print(format('How to Select Elements from Numpy Array', '*^52'))    
    
    # Load library
    import numpy as np

    # Create row vector
    vector = np.array([1, 2, 3, 4, 5, 6])
    # Select second element
    print()
    print(vector[1])

    # Create matrix
    matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])
    # Select second row, second column
    print()
    print(matrix[1,1])

    # Create Tensor
    tensor = np.array([
                    [[[1, 1], [1, 1]], [[2, 2], [2, 2]]],
                    [[[3, 3], [3, 3]], [[4, 4], [4, 4]]]
                  ])
    # Select second element of each of the three dimensions
    print()
    print(tensor[1,1,1])
Kickstarter_Example_3()    
    

## How to Reshape a Numpy Array or Matrix
def Kickstarter_Example_4(): 
    print()
    print(format('How to Reshape a Numpy Array', '*^52'))    
    # Load library
    import numpy as np
    # Create a 4x3 matrix
    matrix = np.array([[1, 2, 3],
                       [4, 5, 6],
                       [7, 8, 9],
                       [10, 11, 12]])
    # Reshape matrix into 2x6 matrix
    print()
    print(matrix.reshape(2, 6))
    print()
    print(matrix.reshape(3, 4))
    print()
    print(matrix.reshape(6, 2))
Kickstarter_Example_4()

## How to Convert a Dictionary into a Matrix or nArray
def Kickstarter_Example_5(): 
    print()
    print(format('How to Convert a Dictionary into a Matrix or ndArray', 
                 '*^72'))    
    # Load library
    from sklearn.feature_extraction import DictVectorizer
    # Our dictionary of data
    data_dict = [{'Apple': 2, 'Orange': 4},
                 {'Apple': 4, 'Orange': 3},
                 {'Apple': 1, 'Banana': 2},
                 {'Apple': 2, 'Banana': 2}]
    print()
    print(data_dict)
    # Create DictVectorizer object
    dictvectorizer = DictVectorizer(sparse=False)
    # Convert dictionary into feature matrix
    features = dictvectorizer.fit_transform(data_dict)
    # View feature matrix
    print()
    print(features)
    # View feature matrix column names
    dictvectorizer.get_feature_names()
Kickstarter_Example_5()

## How to Invert a Matrix or nArray
def Kickstarter_Example_6(): 
    print()
    print(format('How to Invert a Matrix or ndArray', '*^72'))    
    # Load library
    import numpy as np
    # Create a 3x3 matrix
    matrix = np.array([[1, 2, 3],
                       [4, 5, 6],
                       [7, 8, 9]])
    # Calculate inverse of matrix
    Im = np.linalg.inv(matrix)
    print()
    print(Im)
Kickstarter_Example_6()

## How to Calculate Trace of a Matrix
def Kickstarter_Example_7(): 
    print()
    print(format('How to Calculate Trace of a Matrix', '*^72'))    
    # Load library
    import numpy as np
    # Create matrix
    matrix = np.array([[1, 2, 3],
                       [4, 5, 6],
                       [7, 8, 9],
                       [10, 11, 12]])
    # Calculate the tracre of the matrix
    print()
    print('Calculate the tracre of the matrix: ', 
           matrix.diagonal().sum())
Kickstarter_Example_7()

## How to get Diagonal of a Matrix
def Kickstarter_Example_8(): 
    print()
    print(format('How to get Diagonal of a Matrix', '*^72'))    
    # Load library
    import numpy as np
    # Create matrix
    matrix = np.array([[1, 2, 3, 23],
                       [4, 5, 6, 25],
                       [7, 8, 9, 28],
                       [10, 11, 12, 41]])
    # Return diagonal elements
    print()
    print(matrix.diagonal())
    # Calculate the tracre of the matrix
    print()
    print(matrix.diagonal().sum())
Kickstarter_Example_8()

## How to Calculate Determinant of a Matrix or ndArray
def Kickstarter_Example_9(): 
    print()
    print(format('How to Calculate Determinant of a Matrix or ndArray'
                 , '*^72'))    
    # Load library
    import numpy as np
    # Create matrix
    matrixA = np.array([[1, 2, 3, 23],
                       [4, 5, 6, 25],
                       [7, 8, 9, 28],
                       [10, 11, 12, 41]])

    matrixB = np.array([[2, 3, 4],
                       [5, 6, 9],
                       [7, 8, 1]])        
    # Return determinant of matrix
    print(); print(np.linalg.det(matrixA))
    print(); print(np.linalg.det(matrixB))
Kickstarter_Example_9()



## How to Calculate Mean, Variance and Std a Matrix or ndArray
def Kickstarter_Example_11(): 
    print()
    print(format('How to Calculate Mean, Variance and Std a Matrix or ndArray'
                 ,'*^72'))    
    # Load library
    import numpy as np
    # Create matrix
    matrixA = np.array([[1, 2, 3, 23],
                       [4, 5, 6, 25],
                       [7, 8, 9, 28],
                       [10, 11, 12, 41]])
    # Return median, mean, variance and std
    print(); print("Median: ", np.median(matrixA))
    print(); print("Mean: ", np.mean(matrixA))
    print(); print("Variance: ", np.var(matrixA))
    print(); print("Standrad Dev: ", np.std(matrixA))
Kickstarter_Example_11()

## How to find the Rank of a Matrix
def Kickstarter_Example_12(): 
    print()
    print(format('How to find the Rank of a Matrix','*^72'))    
    # Load library
    import numpy as np
    # Create matrix
    matrixA = np.array([[1, 2, 3, 23],
                       [4, 5, 6, 25],
                       [7, 8, 9, 28],
                       [10, 11, 12, 41]])
    # Return the Rank of a Matrix
    print(); print("The Rank of a Matrix: ", 
                    np.linalg.matrix_rank(matrixA))
Kickstarter_Example_12()

## How to find Maximum and Minimum values in a Matrix
def Kickstarter_Example_13(): 
    print()
    print(format('How to find Maximum and Minimum values in a Matrix',
                 '*^72'))    
    # Load library
    import numpy as np
    # Create matrix
    matrixA = np.array([[1, 2, 3, 23],
                       [4, 5, 6, 25],
                       [7, 8, 9, 28],
                       [10, 11, 12, 41]])
    # Return maximum element
    print(); print(np.max(matrixA))
    # Return minimum element
    print(); print(np.min(matrixA))
    # Find the maximum element in each column
    print(); print(np.max(matrixA, axis=0))
    # Find the maximum element in each row
    print(); print(np.max(matrixA, axis=1))
Kickstarter_Example_13()

## How to calculate dot product of two vectors
def Kickstarter_Example_14(): 
    print()
    print(format('How to calculate dot product of two vectors','*^72'))    
    # Load library
    import numpy as np
    # Create two vectors
    vectorA = np.array([1,2,3])
    vectorB = np.array([4,5,6])
    # Calculate Dot Product (Method 1)
    print(); print(np.dot(vectorA, vectorB))
    # Calculate Dot Product (Method 2)
    print(); print(vectorA @ vectorB)
Kickstarter_Example_14()

## How to calculate dot product of two matrices
def Kickstarter_Example_14(): 
    print()
    print(format('How to calculate dot product of two matrices','*^72'))    
    # Load library
    import numpy as np
    # Create two vectors
    matrixA = np.array([[2, 3, 23],
                       [5, 6, 25],
                       [8, 9, 28]])
    matrixB = np.array([[1, 2, 3],
                       [4, 5, 6],
                       [7, 8, 9]])
    # Calculate Dot Product (Method 1)
    print(); print(np.dot(matrixA, matrixB))
    # Calculate Dot Product (Method 2)
    print(); print(matrixA @ matrixB)
Kickstarter_Example_14()

## How to describe a matrix
def Kickstarter_Example_14(): 
    print()
    print(format('How to describe a matrix','*^72'))    
    # Load library
    import numpy as np
    # Create a matrix
    matrixA = np.array([[2, 3, 23],
                       [5, 6, 25],
                       [8, 9, 28]])
    # View number of rows and columns
    print(); print("Shape: ", matrixA.shape)
    # View number of elements (rows * columns)
    print(); print("Size: ", matrixA.size)
    # View number of dimensions
    print(); print("Dimention: ", matrixA.ndim)
Kickstarter_Example_14()

## How to ADD numerical value to each electment of a matrix
def Kickstarter_Example_15(): 
    print()
    print(format('How to add something to each electment of a matrix',
                 '*^72'))    
    # Load library
    import numpy as np
    # Create two vectors
    matrixA = np.array([[2, 3, 23],
                       [5, 6, 25],
                       [8, 9, 28]])
    # Create a function that adds 100 to something
    add_100 = lambda i: i + 100
    # Create a vectorized function
    vectorized_add_100 = np.vectorize(add_100)
    # Apply function to all elements in matrix
    print(); print(vectorized_add_100(matrixA))
Kickstarter_Example_15()

## How to SUBTRACT numerical value to each electment of a matrix
def Kickstarter_Example_16(): 
    print()
    print(format('How to subtract something to each electment of a matrix',
                 '*^72'))    
    # Load library
    import numpy as np
    # Create two vectors
    matrixA = np.array([[2, 3, 23],
                       [5, 6, 25],
                       [8, 9, 28]])
    # Create a function that adds 100 to something
    add_100 = lambda i: i - 15
    # Create a vectorized function
    vectorized_add_100 = np.vectorize(add_100)
    # Apply function to all elements in matrix
    print(); print(vectorized_add_100(matrixA))
Kickstarter_Example_16()

## How to MULTIPLY numerical value to each electment of a matrix
def Kickstarter_Example_17(): 
    print()
    print(format('How to multiply something to each electment of a matrix',
                 '*^72'))    
    # Load library
    import numpy as np
    # Create two vectors
    matrixA = np.array([[2, 3, 23],
                       [5, 6, 25],
                       [8, 9, 28]])
    # Create a function that adds 100 to something
    add_100 = lambda i: i * 9
    # Create a vectorized function
    vectorized_add_100 = np.vectorize(add_100)
    # Apply function to all elements in matrix
    print(); print(vectorized_add_100(matrixA))
Kickstarter_Example_17()

## How to Divide each electment of a matrix by a numerical value
def Kickstarter_Example_18(): 
    print()
    print(format('How to divide each electment of a matrix by a numerical value',
                 '*^72'))    
    # Load library
    import numpy as np
    # Create two vectors
    matrixA = np.array([[2, 3, 23],
                       [5, 6, 25],
                       [8, 9, 28]])
    # Create a function that adds 100 to something
    add_100 = lambda i: i / 9
    # Create a vectorized function
    vectorized_add_100 = np.vectorize(add_100)
    # Apply function to all elements in matrix
    print(); print(vectorized_add_100(matrixA))
Kickstarter_Example_18()

## How to adding and subtracting between matrices
def Kickstarter_Example_19(): 
    print()
    print(format('How to adding and subtracting between matrices', '*^72'))    
    # Load library
    import numpy as np
    # Create two vectors
    matrixA = np.array([[2, 3, 23],
                       [5, 6, 25],
                       [8, 9, 28]])
    matrixB = np.array([[1, 2, 3],
                       [4, 5, 6],
                       [7, 8, 9]])
    # Add two matrices
    print(); print(np.add(matrixA, matrixB))
    # Subtract two matrices
    print(); print(np.subtract(matrixA, matrixB))
Kickstarter_Example_19()

## How to load features from a Dictionary in python
def Kickstarter_Example_20(): 
    print()
    print(format('How to load features from a Dictionary in python', '*^72'))    
    # Load library
    from sklearn.feature_extraction import DictVectorizer
    # Create A Dictionary
    employee = [{'name': 'Steve Miller', 'age': 33., 'dept': 'Analytics'},
                {'name': 'Lyndon Jones', 'age': 42., 'dept': 'Finance'},
                {'name': 'Baxter Morth', 'age': 37., 'dept': 'Marketing'},
                {'name': 'Mathew Scott', 'age': 32., 'dept': 'Business'}]
    # Convert Dictionary To Feature Matrix
    vec = DictVectorizer()
    # Fit then transform the dictionary with vec, then output an array
    print(); 
    print("Feature Matrix: "); print(vec.fit_transform(employee).toarray())
    # View Feature Names
    print()
    print("Feature Name: "); print(vec.get_feature_names())
Kickstarter_Example_20()

## How to load sklearn Boston Housing data 
def Kickstarter_Example_21(): 
    print()
    print(format('How to load sklearn Boston housing data', '*^72'))    
    # Load libraries
    from sklearn import datasets
    # Load Boston Housing Dataset
    boston = datasets.load_boston()
    # Create feature matrix
    X = boston.data
    print(); print(X.shape);
    #print(X)
    # Create target vector
    y = boston.target
    print(); print(y.shape); 
    #print(y)
Kickstarter_Example_21()

## How to Create simulated data for regression in Python 
def Kickstarter_Example_22(): 
    print()
    print(format('How to Create simulated data for regression in Python', '*^82'))    
    # Load libraries
    import pandas as pd
    from sklearn.datasets import make_regression
    # Create Simulated Data
    # Generate fetures, outputs, and true coefficient of 100 samples,
    features, output, coef = make_regression(n_samples = 100, n_features = 3, 
                                n_informative = 3, n_targets = 1,
                                noise = 0.0, coef = True)
    # View Simulated Data
    # View the features of the first five rows
    print()
    print(pd.DataFrame(features, columns=['Feature 1', 'Feature 2', 'Feature 3']).head())
    # View the output of the first five rows
    print()
    print(pd.DataFrame(output, columns=['Target']).head())
    # View the actual, true coefficients used to generate the data
    print()
    print(pd.DataFrame(coef, columns=['True Coefficient Values']))
Kickstarter_Example_22()

## How to Create simulated data for classification in Python 
def Kickstarter_Example_23(): 
    print()
    print(format('How to Create simulated data for classification in Python', '*^82'))    
    # Load libraries
    from sklearn.datasets import make_classification
    import pandas as pd
    # Create Simulated Data
    # Create a simulated feature matrix and output vector with 100 samples,
    features, output = make_classification(n_samples = 100,
                                       n_features = 10,
                                       n_informative = 10,
                                       n_redundant = 0,
                                       n_classes = 3,
                                       weights = [.2, .3, .8])
    # View the first five observations and their 10 features
    print()
    print("Feature Matrix: "); 
    print(pd.DataFrame(features, columns=['Feature 1', 'Feature 2', 'Feature 3',
         'Feature 4', 'Feature 5', 'Feature 6', 'Feature 7', 'Feature 8', 'Feature 9',
         'Feature 10']).head())
    # View the first five observation's classes
    print()
    print("Target Class: "); 
    print(pd.DataFrame(output, columns=['TargetClass']).head())
Kickstarter_Example_23()

## How to Create simulated data for clustering in Python 
def Kickstarter_Example_24(): 
    print()
    print(format('How to Create simulated data for clustering in Python', '*^82'))    
    # Load libraries
    from sklearn.datasets import make_blobs
    import matplotlib.pyplot as plt
    import pandas as pd
    # Make the features (X) and output (y) with 200 samples,
    features, clusters = make_blobs(n_samples = 2000,
                  n_features = 10, centers = 5,
                  # with .5 cluster standard deviation,
                  cluster_std = 0.4,
                  shuffle = True)
    # View the first five observations and their 10 features
    print()
    print("Feature Matrix: "); 
    print(pd.DataFrame(features, columns=['Feature 1', 'Feature 2', 'Feature 3',
         'Feature 4', 'Feature 5', 'Feature 6', 'Feature 7', 'Feature 8', 
         'Feature 9', 'Feature 10']).head())    
    # Create a scatterplot of the first and second features
    plt.scatter(features[:,0], features[:,1])
    # Show the scatterplot
    plt.show()
Kickstarter_Example_24()

## How to prepare a machine leaning workflow in Python 
def Kickstarter_Example_25(): 
    print()
    print(format('How to prepare a machine leaning workflow in Python', '*^82'))    

    import warnings
    warnings.filterwarnings("ignore")
    # Load libraries
    from sklearn import datasets
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Perceptron
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, confusion_matrix
    # Load the iris dataset
    iris = datasets.load_iris()
    # Create our X and y data
    X = iris.data
    y = iris.target
    # Split the data into 70% training data and 30% test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    # Preprocess The X Data By Scaling
    sc = StandardScaler(with_mean=True, with_std=True)
    sc.fit(X_train)
    # Apply the scaler to the X training data
    X_train_std = sc.transform(X_train)
    # Apply the SAME scaler to the X test data
    X_test_std = sc.transform(X_test)
    #Train A Perceptron Learner
    ppn = Perceptron(alpha=0.0001, class_weight=None, eta0=0.1, 
                     fit_intercept=True, n_iter=40, n_jobs=4, 
                     penalty=None, random_state=0, shuffle=True,
                     verbose=0, warm_start=False)
    # Train the perceptron
    ppn.fit(X_train_std, y_train)
    # Apply The Trained Learner To Test Data
    y_pred = ppn.predict(X_test_std)
    # Compare The Predicted Y With The True Y
    # View the predicted y test data
    print(); print("y_pred: ", y_pred)
    # View the true y test data
    print(); print("y_test: ", y_test)
    # Examine Accuracy Metric
    print(); print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
    print(); print('Comfusion Matrix:\n', confusion_matrix(y_test, y_pred))    
Kickstarter_Example_25()


## How to convert Categorical features to Numerical Features in Python 
def Kickstarter_Example_26(): 
    print()
    print(format('How to convert Categorical features to Numerical Features in Python', 
                 '*^82'))    
    import warnings
    warnings.filterwarnings("ignore")
    # Load libraries
    from sklearn import preprocessing
    import pandas as pd
    #Create DataFrame
    raw_data = {'patient': [1, 1, 1, 2, 2],
                'obs': [1, 2, 3, 1, 2],
                'treatment': [0, 1, 0, 1, 0],
                'score': ['strong', 'weak', 'normal', 'weak', 'strong']}
    df = pd.DataFrame(raw_data, columns = ['patient', 'obs', 'treatment', 'score'])
    # Fit The Label Encoder
    # Create a label (category) encoder object
    le = preprocessing.LabelEncoder()
    # Fit the encoder to the pandas column
    le.fit(df['score'])
    # View The Labels
    print(); print(list(le.classes_))
    # Transform Categories Into Integers
    # Apply the fitted encoder to the pandas column
    print(); print(le.transform(df['score']))
    # Transform Integers Into Categories
    # Convert some integers into their category names
    print(); print(list(le.inverse_transform([2, 2, 1, 0, 1, 2])))
Kickstarter_Example_26()

## How to impute missing class labels in Python 
def Kickstarter_Example_27(): 
    print()
    print(format('How to impute missing class labels in Python', '*^82'))    
    import warnings
    warnings.filterwarnings("ignore")
    # Load libraries
    import numpy as np
    from sklearn.preprocessing import Imputer
    # Create Feature Matrix With Missing Values
    X = np.array([[2,       2.10, 1.45], 
                  [1,       1.18, 1.33], 
                  [2,       1.22, 1.27],
                  [0,       -0.21, -1.19],
                  [np.nan,  0.87, 1.31],
                  [np.nan, -0.67, -0.22]])
    # Create Imputer object
    imputer = Imputer(strategy='most_frequent', axis=0)
    # Fill missing values with most frequent class
    print(); print(X)
    print(); print(imputer.fit_transform(X))
Kickstarter_Example_27()

## How to impute missing class labels using nearest neighbours in Python 
def Kickstarter_Example_28(): 
    print()
    print(format('How to impute missing class labels using nearest neighbours in Python', '*^82'))    
    import warnings
    warnings.filterwarnings("ignore")
    # Load libraries
    import numpy as np
    from sklearn.neighbors import KNeighborsClassifier
    # Create Feature Matrix
    # Create feature matrix with categorical feature
    X = np.array([[0, 2.10, 1.45], 
                  [2, 1.18, 1.33], 
                  [0, 1.22, 1.27],
                  [1, 1.32, 1.97],
                  [1, -0.21, -1.19]])
    # Create Feature Matrix With Missing Values
    # Create feature matrix with missing values in the categorical feature
    X_with_nan = np.array([[np.nan, 0.87, 1.31], 
                           [np.nan, 0.37, 1.91],
                           [np.nan, 0.54, 1.27],
                           [np.nan, -0.67, -0.22]])
    # Train k-Nearest Neighbor Classifier
    clf = KNeighborsClassifier(3, weights='distance')
    trained_model = clf.fit(X[:,1:], X[:,0])
    # Predict missing values' class
    imputed_values = trained_model.predict(X_with_nan[:,1:])
    print(); print(imputed_values)
    # Join column of predicted class with their other features
    X_with_imputed = np.hstack((imputed_values.reshape(-1,1), X_with_nan[:,1:]))
    print(); print(X_with_imputed)
    # Join two feature matrices
    print(); print(np.vstack((X_with_imputed, X)))
Kickstarter_Example_28()

## How to delete instances with missing values in Python 
def Kickstarter_Example_29(): 
    print()
    print(format('How to delete instances with missing values in Python', '*^82'))    
    import warnings
    warnings.filterwarnings("ignore")
    # Load libraries
    import numpy as np
    # Create feature matrix
    X = np.array([[1.1, 11.1], 
                  [2.2, 22.2], 
                  [3.3, 33.3], 
                  [4.4, 44.4], 
                  [np.nan, 55]])
    # Remove observations with missing values
    X = X[~np.isnan(X).any(axis=1)]
    print(); print(X)
Kickstarter_Example_29()

## How to find outliers in Python 
def Kickstarter_Example_30(): 
    print()
    print(format('How to find outliers in Python', '*^82'))    
    import warnings
    warnings.filterwarnings("ignore")
    # Load libraries
    from sklearn.covariance import EllipticEnvelope
    from sklearn.datasets import make_blobs
    import matplotlib.pyplot as plt
    # Create simulated data
    X, _ = make_blobs(n_samples = 100,
                      n_features = 20,
                      centers = 7, 
                      cluster_std = 1.1,
                      shuffle = True,
                      random_state = 42)
    # Detect Outliers
    # Create detector
    outlier_detector = EllipticEnvelope(contamination=.1)
    # Fit detector
    outlier_detector.fit(X)
    # Predict outliers
    print(); print(X)
    print(); print(outlier_detector.predict(X))
    plt.scatter(X[:,0], X[:,1])
    # Show the scatterplot
    plt.show()
Kickstarter_Example_30()

## How to encode ordinal categorical features in Python 
def Kickstarter_Example_31(): 
    print()
    print(format('How to encode ordinal categorical features in Python', '*^82'))    
    import warnings
    warnings.filterwarnings("ignore")
    # Load library
    import pandas as pd
    # Create features
    df = pd.DataFrame({'Score': ['Low', 'Low', 'Medium', 'Medium', 'High']})
    # View data frame
    print(); print(df)
    # Create Scale Map
    scale_mapper = {'Low':1, 'Medium':2, 'High':3}
    # Map feature values to scale
    df['Scale'] = df['Score'].replace(scale_mapper)
    # View data frame
    print(); print(df)
Kickstarter_Example_31()

## How to deal with imbalance classes with downsampling in Python 
def Kickstarter_Example_32(): 
    print()
    print(format('How to deal with imbalance classes with downsampling in Python', '*^82'))    
    import warnings
    warnings.filterwarnings("ignore")
    # Load libraries
    import numpy as np
    from sklearn.datasets import load_iris
    # Load iris data
    iris = load_iris()
    # Create feature matrix
    X = iris.data
    # Create target vector
    y = iris.target
    #Make Iris Dataset Imbalanced # Remove first 40 observations
    X = X[40:,:]
    y = y[40:]
    # Create binary target vector indicating if class 0
    y = np.where((y == 0), 0, 1)
    # Look at the imbalanced target vector
    print(); print("Look at the imbalanced target vector:\n", y)
    # Downsample Majority Class To Match Minority Class
    # Indicies of each class' observations
    i_class0 = np.where(y == 0)[0]
    i_class1 = np.where(y == 1)[0]
    # Number of observations in each class
    n_class0 = len(i_class0); print(); print("n_class0: ", n_class0)
    n_class1 = len(i_class1); print(); print("n_class1: ", n_class1)
    # For every observation of class 0, randomly sample from class 1 without replacement
    i_class1_downsampled = np.random.choice(i_class1, size=n_class0, replace=False)
    # Join together class 0's target vector with the downsampled class 1's target vector
    print(); print(np.hstack((y[i_class0], y[i_class1_downsampled])))
Kickstarter_Example_32()

## How to deal with imbalance classes with upsampling in Python 
def Kickstarter_Example_33(): 
    print()
    print(format('How to deal with imbalance classes with upsampling in Python', '*^82'))    
    import warnings
    warnings.filterwarnings("ignore")
    # Load libraries
    import numpy as np
    from sklearn.datasets import load_iris
    # Load iris data
    iris = load_iris()
    # Create feature matrix
    X = iris.data
    # Create target vector
    y = iris.target
    #Make Iris Dataset Imbalanced # Remove first 40 observations
    X = X[40:,:]
    y = y[40:]
    # Create binary target vector indicating if class 0
    y = np.where((y == 0), 0, 1)
    # Look at the imbalanced target vector
    print(); print("Look at the imbalanced target vector:\n", y)
    # Downsample Majority Class To Match Minority Class
    # Indicies of each class' observations
    i_class0 = np.where(y == 0)[0]
    i_class1 = np.where(y == 1)[0]
    # Number of observations in each class
    n_class0 = len(i_class0); print(); print("n_class0: ", n_class0)
    n_class1 = len(i_class1); print(); print("n_class1: ", n_class1)
    # For every observation of class 1, randomly sample from class 0 with replacement
    i_class0_upsampled = np.random.choice(i_class0, size=n_class1, replace=True)
    # Join together class 1's target vector with the upsampled class 0's target vector
    print(); print(np.hstack((y[i_class0_upsampled], y[i_class1])))
Kickstarter_Example_33()

## How to deal with outliers in Python 
def Kickstarter_Example_34(): 
    print()
    print(format('How to deal with outliers in Python ', '*^82'))    
    import warnings
    warnings.filterwarnings("ignore")
    # Load library
    import numpy as np
    import pandas as pd
    # Create DataFrame
    houses = pd.DataFrame()
    houses['Price'] = [534433, 392333, 293222, 4322032]
    houses['Bathrooms'] = [2, 3.5, 2, 116]
    houses['Square_Feet'] = [1500, 2500, 1500, 48000]
    print(); print(houses)
    
    # Outlier Handling Option 1: Drop
    # Drop observations greater than some value
    h = houses[houses['Bathrooms'] < 20]
    print(); print(h)
    # Outlier Handling Option 2: Mark
    # Create feature based on boolean condition
    houses['Outlier'] = np.where(houses['Bathrooms'] < 20, 0, 1)
    # Show data
    print(); print(houses)
    
    # Outlier Handling Option 3: Rescale
    # Log feature
    houses['Log_Of_Square_Feet'] = [np.log(x) for x in houses['Square_Feet']]
    # Show data
    print(); print(houses)
Kickstarter_Example_34()

## How to impute missing values with means in Python 
def Kickstarter_Example_35(): 
    print()
    print(format('How to impute missing values with means in Python', '*^82'))    
    import warnings
    warnings.filterwarnings("ignore")
    # load libraries
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import Imputer
    # Create an empty dataset
    df = pd.DataFrame()
    # Create two variables called x0 and x1. Make the first value of x1 a missing value
    df['V0'] = [0.3051,0.4949,0.6974,0.3769,0.2231,
                0.341,0.4436,0.5897,0.6308,0.5]
    df['V1'] = [np.nan,np.nan,0.2615,0.5846,0.4615,
                0.8308,0.4962,np.nan,0.5346,0.6731]
    # View the dataset
    print(); print(df)
    # Create an imputer object that looks for 'Nan' values, 
    # then replaces them with the mean value of the feature by columns (axis=0)
    mean_imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
    # Train the imputor on the df dataset
    mean_imputer = mean_imputer.fit(df)
    # Apply the imputer to the df dataset
    imputed_df = mean_imputer.transform(df.values)
    # View the data
    print(); print(imputed_df)
Kickstarter_Example_35()

## One hot Encoding with multiple labels in Python 
def Kickstarter_Example_36(): 
    print()
    print(format('How to One hot Encode with multiple labels in Python', '*^82'))    
    import warnings
    warnings.filterwarnings("ignore")
    # Load libraries
    from sklearn.preprocessing import MultiLabelBinarizer
    # Create NumPy array
    y = [('Texas', 'Florida'), 
         ('California', 'Alabama'), 
         ('Texas', 'Florida'), 
         ('Delware', 'Florida'), 
         ('Texas', 'Alabama')]
    # Create MultiLabelBinarizer object
    one_hot = MultiLabelBinarizer()
    # One-hot encode data
    print(); print(one_hot.fit_transform(y))
    # View Column Headers
    # View classes
    print(); print(one_hot.classes_)
Kickstarter_Example_36()


## One hot Encoding with nominal categorical features in Python 
def Kickstarter_Example_37(): 
    print()
    print(format('How to One hot Encode with nominal categorical features in Python', '*^82'))    
    import warnings
    warnings.filterwarnings("ignore")
    # Load libraries
    import numpy as np
    from sklearn.preprocessing import LabelBinarizer
    # Create Data With One Class Label
    # Create NumPy array
    x = np.array([['Texas'], 
                  ['California'], 
                  ['Texas'], 
                  ['Delaware'], 
                  ['Texas']])
    # One-hot Encode Data (Method 1)
    # Create LabelBinzarizer object
    one_hot = LabelBinarizer()
    # One-hot encode data
    print(); print(one_hot.fit_transform(x))
    # View Column Headers
    # View classes
    print(); print(one_hot.classes_)
Kickstarter_Example_37()

## How to process categorical features in Python 
def Kickstarter_Example_38(): 
    print()
    print(format('How to process categorical features in Python', '*^82'))    
    import warnings
    warnings.filterwarnings("ignore")
    # load libraries
    from sklearn import preprocessing
#    from sklearn.pipeline import Pipeline
    import pandas as pd
    # Create Data
    raw_data = {'first_name': ['Jason', 'Molly', 'Tina', 'Jake', 'Amy'], 
                'last_name': ['Miller', 'Jacobson', 'Ali', 'Milner', 'Cooze'], 
                'age': [42, 52, 36, 24, 73], 
                'city': ['San Francisco', 'Baltimore', 'Miami', 'Douglas', 'Boston']}
    df = pd.DataFrame(raw_data, columns = ['first_name', 'last_name', 'age', 'city'])
    print(); print(df)
    # Convert Nominal Categorical Feature Into Dummy Variables Using Pandas
    # Create dummy variables for every unique category in df.city
    print(); print(pd.get_dummies(df["city"]))
    
    # Convert Nominal Categorical Data Into Dummy (OneHot) Features Using Scikit
    # Convert strings categorical names to integers
    integerized_data = preprocessing.LabelEncoder().fit_transform(df["city"])
    # View data
    print(); print(integerized_data)
    # Convert integer categorical representations to OneHot encodings
    output = preprocessing.OneHotEncoder().fit_transform(integerized_data.reshape(-1,1)).toarray()
    print(); print(output)
Kickstarter_Example_38()

## How to rescale features in Python 
def Kickstarter_Example_39(): 
    print()
    print(format('How to rescale features in Python', '*^82'))    
    import warnings
    warnings.filterwarnings("ignore")
    # load libraries
    from sklearn import preprocessing
    import numpy as np
    # Create feature
    x = np.array([[-500.5], 
                  [-100.1], 
                  [0], 
                  [100.1], 
                  [900.9]])
    # Rescale Feature Using Min-Max
    # Create scaler
    minmax_scale = preprocessing.MinMaxScaler(feature_range=(0, 1))
    # Scale feature
    x_scale = minmax_scale.fit_transform(x)
    # Show feature
    print(); print(x)    
    print(); print(x_scale)
Kickstarter_Example_39()


## How to standarise features in Python 
def Kickstarter_Example_40(): 
    print()
    print(format('How to standarise features in Python', '*^82'))    
    import warnings
    warnings.filterwarnings("ignore")
    # Load libraries
    from sklearn import preprocessing
    import numpy as np
    # Create feature
    x = np.array([[-500.5], 
                  [-100.1], 
                  [0], 
                  [100.1], 
                  [900.9]])
    # Standardize Feature
    # Create scaler
    scaler = preprocessing.StandardScaler()
    # Transform the feature
    standardized_x = scaler.fit_transform(x)
    # Show feature
    print(); print(x)
    print(); print(standardized_x)
Kickstarter_Example_40()

## How to standarise IRIS Data in Python 
def Kickstarter_Example_41(): 
    print()
    print(format('How to standarise IRIS Data in Python', '*^82'))    
    import warnings
    warnings.filterwarnings("ignore")
    
    from sklearn import datasets
    from sklearn.cross_validation import train_test_split
    from sklearn.preprocessing import StandardScaler
    
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, 
                                                        random_state=42)
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

    print(); print(X_train[0:5])
    print(); print(X_train_std[0:5])
    print(); print(X_test[0:5])
    print(); print(X_test_std[0:5])
Kickstarter_Example_41()

## How to split DateTime Data to create multiple feature in Python 
def Kickstarter_Example_42(): 
    print()
    print(format('How to split DateTime Data to create multiple feature in Python', '*^82'))    
    import warnings
    warnings.filterwarnings("ignore")
    
    # Load library
    import pandas as pd
    # Create data frame
    df = pd.DataFrame()
    # Create dates
    df['date'] = pd.date_range('1/1/2018', periods=5, freq='M')
    print(df)
    # Create features for year, month, day, hour, and minute
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['hour'] = df['date'].dt.hour
    df['minute'] = df['date'].dt.minute

    # Show three rows
    print()
    print(df.head(3))
Kickstarter_Example_42()

## How to claculate difference between Dates in Python 
def Kickstarter_Example_42(): 
    print()
    print(format('How to claculate difference between Dates in Python',
                 '*^82'))    
    import warnings
    warnings.filterwarnings("ignore")

    # Load library
    import pandas as pd
    # Create data frame
    df = pd.DataFrame() 
    # Create two datetime features
    df['Date1'] = [pd.Timestamp('01-01-2017'), 
                     pd.Timestamp('01-04-2017')]
    df['Date2']    = [pd.Timestamp('01-01-2017'), 
                     pd.Timestamp('01-06-2017')]
    # Calculate Difference (Method 1)
    print(df['Date2'] - df['Date1'])
    # Calculate Difference (Method 2)
    print(pd.Series(delta.days for delta in (df['Date2'] - df['Date1'])))
Kickstarter_Example_42()

## How to convert Strings to DateTimes in Python 
def Kickstarter_Example_43(): 
    print()
    print(format('How to convert Strings to DateTimes in Python',
                 '*^82'))    
    import warnings
    warnings.filterwarnings("ignore")
    
    # Load libraries
    import numpy as np
    import pandas as pd
    # Create strings
    date_strings = np.array(['01-01-2015 11:35 PM',
                             '23-02-2016 12:01 AM',
                             '26-12-2017 09:09 PM'])
    print()
    print(date_strings)
    # Convert to datetimes
    print()
    print([pd.to_datetime(date, 
                    format="%d-%m-%Y %I:%M %p", 
                    errors="coerce") for date in date_strings])
Kickstarter_Example_43()

## How to encode Days of a week in Python 
def Kickstarter_Example_44(): 
    print()
    print(format('How to encode Days of a week in Python',
                 '*^82'))    
    import warnings
    warnings.filterwarnings("ignore")
    # Load library
    import pandas as pd
    # Create dates
    dates = pd.Series(pd.date_range('11/9/2018', periods=3, freq='M'))
    # View data
    print()
    print(dates)
    # Show days of the week
    print()
    print(dates.dt.weekday_name)
Kickstarter_Example_44()

## How to deal with missing values in a Timeseries in Python 
def Kickstarter_Example_45(): 
    print()
    print(format('How to deal with missing values in a Timeseries in Python',
                 '*^82'))    
    import warnings
    warnings.filterwarnings("ignore")

    # Load libraries
    import pandas as pd
    import numpy as np
    # Create date
    time_index = pd.date_range('28/03/2017', periods=5, freq='M')
    # Create data frame, set index
    df = pd.DataFrame(index=time_index); print(df)
    # Create feature with a gap of missing values
    df['Sales'] = [1.0,2.0,np.nan,np.nan,5.0]; print(); print(df)
    # Interpolate missing values
    df1= df.interpolate(); print(); print(df1)    
    # Forward-fill Missing Values
    df2 = df.ffill(); print(); print(df2)    
    # Backfill Missing Values
    df3 = df.bfill(); print(); print(df3)    
    # Interpolate Missing Values But Only Up One Value
    df4 = df.interpolate(limit=1, limit_direction='forward'); print(); print(df4)
    # Interpolate Missing Values But Only Up Two Values
    df5 = df.interpolate(limit=2, limit_direction='forward'); print(); print(df5)    
Kickstarter_Example_45()

## How to introduce LAG time in Python 
def Kickstarter_Example_46(): 
    print()
    print(format('How to introduce LAG time in Python','*^82'))    
    import warnings
    warnings.filterwarnings("ignore")

    # Load library
    import pandas as pd
    # Create data frame
    df = pd.DataFrame()
    # Create data
    df['dates'] = pd.date_range('11/11/2016', periods=5, freq='D')
    df['stock_price'] = [1.1,2.2,3.3,4.4,5.5]
    # Lag Time Data By One Row
    df['previous_days_stock_price'] = df['stock_price'].shift(1)
    # Show data frame
    print(); print(df)
    # Lag Time Data By Two Rows
    df['previous_days_stock_price'] = df['stock_price'].shift(2)
    # Show data frame
    print(); print(df)    
Kickstarter_Example_46()

## How to deal with Rolling Tine Window in Python 
def Kickstarter_Example_47(): 
    print()
    print(format('How to deal with Rolling Time Window in Python','*^82'))    
    import warnings
    warnings.filterwarnings("ignore")

    # Load library
    import pandas as pd
    # Create datetimes
    time_index = pd.date_range('01/01/2010', periods=5, freq='M')
    # Create data frame, set index
    df = pd.DataFrame(index=time_index)
    # Create feature
    df['Stock_Price'] = [1,2,3,4,5]
    print(); print(df)
    # Create A Rolling Time Window Of Two Rows
    # Calculate rolling mean
    df1 = df.rolling(window=2).mean()
    print(); print(df1)
    # Identify max value in rolling time window
    df2 = df.rolling(window=2).max()
    print(); print(df2)
Kickstarter_Example_47()

## How to select DateTime within a range in Python 
def Kickstarter_Example_48(): 
    print()
    print(format('How to select DateTime within a range in Python','*^82'))    
    import warnings
    warnings.filterwarnings("ignore")

    # Load library
    import pandas as pd
    # Create data frame
    df = pd.DataFrame()
    # Create datetimes
    df['date'] = pd.date_range('15/12/1999', periods=100000, freq='H')
    # Set index
    df = df.set_index(df['date'])
    # Select observations between two datetimes
    print(); print(df.loc['2002-1-1 01:00:00':'2002-1-1 07:00:00'])
Kickstarter_Example_48()

## How to add padding around string 
def Kickstarter_Example_49(): 
    print()
    print(format('How to add padding around string','*^82'))    
    import warnings
    warnings.filterwarnings("ignore")

    text = 'Mastering in Pytho through Kickstarter_Examples'
    # Add Padding Around Text
    # Add Spaces Of Padding To The Left
    print(); print(format(text, '>75'))
    # Add Spaces Of Padding To The Right
    print(); print(format(text, '<75'))
    # Add Spaces Of Padding On Each Side
    print(); print(format(text, '^75'))
    # Add * Of Padding On Each Side
    print(); print(format(text, '*^75'))
Kickstarter_Example_49()

## How to deal with an Item in a List in Python 
def Kickstarter_Example_50(): 
    print()
    print(format('How to deal with an Item in a List in Python','*^82'))    
    import warnings
    warnings.filterwarnings("ignore")

    # Create a list of sales
    Sales = [482, 93, 392, 920, 813, 199, 374, 237, 244]

    def updated(x): return x + 100
    print(); print(list(map(updated, Sales)))

    salesUpdated = []
    for x in Sales:
        salesUpdated.append(x + 10)
    print(); print(salesUpdated)

    print(); print(list(map((lambda x: x + 100), Sales)))
Kickstarter_Example_50()

## How to do numerical operations in Python using Numpy
def Kickstarter_Example_51(): 
    print()
    print(format('How to do numerical operations in Python using Numpy','*^82'))    
    import warnings
    warnings.filterwarnings("ignore")

    # Load Libraries
    import numpy as np
    # Create an array
    sales = np.array([4352, 233, 3245, 256, 2394])
    print(); print(sales)
    # Mean value of the array
    print(); print(sales.mean())
    # Total amount of deaths
    print(); print(sales.sum())
    # Smallest value in the array
    print(); print(sales.min())
    # Largest value in the array
    print(); print(sales.max())
Kickstarter_Example_51()

## How to compare two dictionaries in Python
def Kickstarter_Example_51a(): 
    print()
    print(format('How to compare two dictionaries in Python','*^82'))    
    import warnings
    warnings.filterwarnings("ignore")

    # Make Two Dictionaries
    importers = {'El Salvador' : 1234,
                 'Nicaragua' : 152,
                 'Spain' : 252
                 }
    exporters = {'Spain' : 252,
                 'Germany' : 251,
                 'Italy' : 1563
                 }
    # Find the intersection
    print(); print(importers.keys() & exporters.keys())
    # Find the difference
    print(); print(importers.keys() - exporters.keys())
    # Find countries in Common
    print(); print(importers.items() & exporters.items())
    # Find All countries 
    print(); print(importers.items() | exporters.items())    
Kickstarter_Example_51a()

## How to use CONTINUE and BREAK statement within a loop in Python
def Kickstarter_Example_52(): 
    print()
    print(format('How to use CONTINUE and BREAK statement within a loop in Python',
                 '*^82'))    
    import warnings
    warnings.filterwarnings("ignore")

    # Import the random module
    import random
    # Create a while loop # set "running" to true
    running = True
    # while running is true
    while running:
        # Create a random integer between 0 and 5
        s = random.randint(0,5)
        # If the integer is less than 3
        if s < 3:
            print(s, ': It is too small, starting again.')
            continue
        # If the integer is 4
        if s == 4:
            running = False
            print('It is 4! Changing running to false')
        # If the integer is 5,
        if s == 5:
            print('It is 5! Breaking Loop!')
            break
Kickstarter_Example_52()

## How to convert STRING to DateTime in Python
def Kickstarter_Example_53(): 
    print()
    print(format('How to convert STRING to DateTime in Python','*^82'))    
    import warnings
    warnings.filterwarnings("ignore")

    # Load Libraries
    from datetime import datetime
    from dateutil.parser import parse
    import pandas as pd
    # Create a string variable with a datetime
    date_start = '2012-03-03'
    # Convert the string to datetime format
    print()
    print(datetime.strptime(date_start, '%Y-%m-%d'))
    # Create a list of strings as dates
    dates = ['7/2/2017', '8/6/2016', '11/13/2015', '5/26/2014', '5/2/2013']
    # Use parse() to attempt to auto-convert common string formats
    print()
    print(parse(date_start))
    
    print()
    print([parse(x) for x in dates])
    # Use parse, but designate that the day is first
    print()
    print(parse(date_start, dayfirst=True))

    # Create a dataframe
    data = {'date': ['2014-05-01 18:47:05.069722', '2014-05-01 18:47:05.119994', 
                     '2014-05-02 18:47:05.178768', '2014-05-02 18:47:05.230071', 
                     '2014-05-02 18:47:05.230071', '2014-05-02 18:47:05.280592', 
                     '2014-05-03 18:47:05.332662', '2014-05-03 18:47:05.385109', 
                     '2014-05-04 18:47:05.436523', '2014-05-04 18:47:05.486877'], 
            'value': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
    df = pd.DataFrame(data, columns = ['date', 'value'])
    print(df.dtypes)
    # Convert df['date'] from string to datetime
    print()
    print(pd.to_datetime(df['date']))
    print(pd.to_datetime(df['date']).dtypes)
Kickstarter_Example_53()

## How to Create and Delete a file in Python
def Kickstarter_Example_54(): 
    print()
    print(format('How to create and Delete a file in Python','*^82'))    
    import warnings
    warnings.filterwarnings("ignore")

    import os
    # Create a file if it doesn't already exist
    with open('file.txt', 'xt') as f:
        # Write to the file
        f.write('This is a New File. Just Created!')
        # Close the connection to the file
        f.close()
    # Open The File And Read It
    with open('file.txt', 'rt') as f:
        # Read the data in the file
        data = f.read()
        # Close the connection to the file
        f.close()
    # View The Contents Of The File
    print(data)

    # Delete The File
    os.remove('file.txt')
Kickstarter_Example_54()

## How to deal with Date & Time Basics in Python
def Kickstarter_Example_55(): 
    print()
    print(format('How to deal with Date & Time Basics in Python','*^82'))    
    import warnings
    warnings.filterwarnings("ignore")

    # Load Libraries
    from datetime import datetime
    from datetime import timedelta
    # Create a variable with the current time
    now = datetime.now()
    print(); print(now)
    # The current year
    print(); print(now.year)
    # The current month
    print(); print(now.month)
    # The current day
    print(); print(now.day)
    # The current hour
    print(); print(now.hour)
    # The current minute
    print(); print(now.minute)
    # The difference between two dates
    delta = datetime(2011, 1, 7) - datetime(2011, 1, 6)
    print(); print(delta)
    # The difference days
    print(); print(delta.days)
    # The difference seconds
    print(); print(delta.seconds)
    # Create a time
    start = datetime(2018, 1, 7)
    # Add twelve days to the time
    print(); print(start + timedelta(12))
Kickstarter_Example_55()


## How to deal with Dictionary Basics in Python
def Kickstarter_Example_56(): 
    print()
    print(format('How to work with Dictionary Basics in Python','*^82'))    
    import warnings
    warnings.filterwarnings("ignore")
    # Build a dictionary via brackets
    unef_org = {'name' : 'UNEF',
                'staff' : 32,
                'url' : 'http://unef.org'}
    # View the variable
    print(); print(unef_org)
    #Build a dict via keys
    who_org = {}
    who_org['name'] = 'WHO'
    who_org['staff'] = '10'
    who_org['url'] = 'https://setscholars.com'
    # View the variable
    print(); print(who_org)
    # Build a dictionary via brackets # Nesting in dictionaries
    unitas_org = {'name' : 'UNITAS',
                  'staff' : 32,
                  'url' : ['https://setscholars.com', 
                           'https://setscholars.info']}
    # View the variable
    print(); print(unitas_org)
    # Index the nested list
    print(); print(unitas_org['url'])
    print(); print(unitas_org['url'][0])
    print(); print(unitas_org['url'][1])
Kickstarter_Example_56()


## How to find MIN, MAX in a Dictionary
def Kickstarter_Example_57(): 
    print()
    print(format('How to find MIN, MAX in a Dictionary','*^82'))    
    import warnings
    warnings.filterwarnings("ignore")
    # Create A Dictionary
    ages = {'John': 21, 'Mike': 52, 'Sarah': 12, 'Bob': 43}
    # Find The Maximum Value Of The Values
    print()
    print('Maximum Value: '); print(max(zip(ages.values(), ages.keys())))
    print()
    print('Maximum Value: '); print(min(zip(ages.values(), ages.keys())))
Kickstarter_Example_57()

## How to define FOR Loop in Python
def Kickstarter_Example_58(): 
    print()
    print(format('How to define FOR Loop in Python','*^82'))    
    import warnings
    warnings.filterwarnings("ignore")

    for i in [10, 20, 30, 40, 50]:
        x = i ** 19
        print(x)
    else: print('All done!')
Kickstarter_Example_58()


## How to define WHILE Loop in Python
def Kickstarter_Example_59(): 
    print()
    print(format('How to define WHILE Loop in Python','*^82'))    
    import warnings
    warnings.filterwarnings("ignore")

    # Load Libraries
    import random
    # Create a variable of the true number of Sales of a Retail event
    sales = 7
    # Create a variable that is denotes if the while loop should keep running
    running = True
    # while running is True
    while running:
        # Create a variable that randomly create a integer between 0 and 10.
        guess = random.randint(0,10)
        # if guess equals number of sales,
        if guess == sales:
            print('Correct!')
            running = False
        # else if guess is lower than slaes
        elif guess < sales:
            print('No, it is higher.')
        else:
            print('No, it is lower')
Kickstarter_Example_59()

## How to create RANDOM Numbers in Python
def Kickstarter_Example_60(): 
    print()
    print(format('How to create RANDOM Numbers in Python','*^82'))    
    import warnings
    warnings.filterwarnings("ignore")
    # Load Libraries
    import numpy as np
    # Generate A Random Number From The Normal Distribution
    print(); print(np.random.normal())
    # Generate Four Random Numbers From The Normal Distribution
    print(); print(np.random.normal(size=14))
    # Generate Four Random Numbers From The Uniform Distribution
    print(); print(np.random.uniform(size=14))
    # Generate Four Random Integers Between 1 and 100
    print(); print(np.random.randint(low=1, high=100, size=14))
Kickstarter_Example_60()


## How to index and slice Numpy arrays in Python
def Kickstarter_Example_61(): 
    print()
    print(format('How to index and slice Numpy arrays in Python','*^82'))    
    import warnings
    warnings.filterwarnings("ignore")

    # Load libraries
    import numpy as np
    # Create an array of battle casualties from the first to the last battle
    battleDeaths = np.array([1245, 2732, 3853, 4824, 5292, 6184, 7282, 81393, 932, 10834])
    # Divide the array of battle deaths into start, middle, and end of the war
    warStart = battleDeaths[0:3]; print('Death from battles at the start of war:', warStart)
    warMiddle = battleDeaths[3:7]; print('Death from battles at the middle of war:', warMiddle)
    warEnd = battleDeaths[7:10]; print('Death from battles at the end of war:', warEnd)
    # Change the battle death numbers from the first battle
    warStart[0] = 11101
    # View that change reflected in the warStart slice of the battleDeaths array
    print(); print(warStart)
    # View that change reflected in (i.e. "broadcasted to) the original battleDeaths array
    print(); print(battleDeaths)
    # Create an array of regiment information
    regimentNames = ['Nighthawks', 'Sky Warriors', 'Rough Riders', 'New Birds']
    regimentNumber = [1, 2, 3, 4]
    regimentSize = [1092, 2039, 3011, 4099]
    regimentCommander = ['Mitchell', 'Blackthorn', 'Baker', 'Miller']
    regiments = np.array([regimentNames, regimentNumber, regimentSize, regimentCommander])
    print(); print(regiments)
    # View the first column of the matrix
    print(); print(regiments[:,0])
    # View the second row of the matrix
    print(); print(regiments[1,])
    # View the top-right quarter of the matrix
    print(); print(regiments[:2,2:])
Kickstarter_Example_61()


## How to iterate a list using if-else in Python
def Kickstarter_Example_62(): 
    print()
    print(format('How to iterate a list using if-else in Python','*^82'))    
    import warnings
    warnings.filterwarnings("ignore")

    # Create some data
    word_list = ['Egypt', 'Watching', 'Eleanor']
    vowels = ['A', 'E', 'I', 'O', 'U']

    # Create a for loop
    # for each item in the word_list,
    for word in word_list:
        # if any word starts with e, where e is vowels,
        if any([word.startswith(e) for e in vowels]):
            print('Is valid')
        else: print('Invalid')
Kickstarter_Example_62()


## How to iterate over multiple lists in Python
def Kickstarter_Example_63(): 
    print()
    print(format('How to iterate over multiple lists in Python','*^82'))    
    import warnings
    warnings.filterwarnings("ignore")

    names = ['James', 'Bob', 'Sarah', 'Marco', 'Nancy', 'Sally']
    ages = [42, 13, 14, 25, 63, 23]
    ids = [1042, 1013, 1014, 1025, 1063, 1023]
    
    #Iterate Over the Lists At Once
    for name, age, iid in zip(names, ages, ids):
        print(); print(name, age, iid)
Kickstarter_Example_63()

## How to use lambda function in Python
def Kickstarter_Example_64(): 
    print()
    print(format('How to use lambda function in Python','*^82'))    
    import warnings
    warnings.filterwarnings("ignore")

    # Create a series, called pipeline, that contains three mini functions
    pipeline = [lambda x: x ** 2 - 1 + 5,
                lambda x: x ** 20 - 2 + 3,
                lambda x: x ** 200 - 1 + 4]
    # For each item in pipeline, run the lambda function with x = 3
    for f in pipeline:
        print(f(2))
Kickstarter_Example_64()

## How to use loop over multiple lists in Python
def Kickstarter_Example_64a(): 
    print()
    print(format('How to use loop over multiple lists in Python','*^82'))    
    import warnings
    warnings.filterwarnings("ignore")

    # Create a list of length 3:
    students = ['John', 'Kamal', 'Roni', 'Jonny', 'Xixi']

    # Create a list of length 4:
    marks = [95,76,87,55]
    # For each element in the first list,
    for stu, mark in zip(students, marks):
        # Display the corresponding index element of the second list:
        print(); print(stu, 'has the following marks:', mark)
Kickstarter_Example_64a()

## How to do common mathematical operations in Python
def Kickstarter_Example_65(): 
    print()
    print(format('How to do common mathematical operations in Python','*^82'))    
    import warnings
    warnings.filterwarnings("ignore")

    # Import the math module
    import math
    #Display the value of pi.
    print(); print(math.pi)
    # Display the value of e.
    print(); print(math.e)
    # Sine, cosine, and tangent
    print(); print(math.sin(2 * math.pi / 180))
    print(); print(math.cos(2 * math.pi / 180))
    print(); print(math.tan(2 * math.pi / 180))
    # Exponent
    print(); print(2 ** 4, pow(2, 4))
    # Absolute value
    print(); print(abs(-20))
    # Summation
    print(); print(sum((1, 2, 3, 4)))
    # Minimum
    print(); print(min(3, 9, 10, 12))
    # Maximum
    print(); print(max(3, 5, 10, 15))
    # Floor
    print(); print(math.floor(2.949))
    # Truncate (drop decimal digits)
    print(); print(math.trunc(32.09292))
    # Truncate (integer conversion)
    print(); print(int(3.292838))
    # Round to an integrer
    print(); print(round(2.943), round(2.499))
    # Round by 2 digits
    round(2.569, 2)
Kickstarter_Example_65()


## How to nested loops in Python
def Kickstarter_Example_66(): 
    print()
    print(format('How to use nested loops in Python','*^82'))    
    import warnings
    warnings.filterwarnings("ignore")

    # Create two lists
    squads = ["1st Squad", '2nd Squad', '3rd Squad']
    regiments = ["51st Regiment", '15th Regiment', '12th Regiment']
    numbers = [25, 50, 75,100]
    # Create a tuple for each regiment in regiments, for each squad in sqauds
    print()
    print([(regiment, squad) for regiment in regiments 
                                 for squad in squads])
    print()
    print([(regiment, squad, numbs) for regiment in regiments 
                                  for squad in squads 
                                      for numbs in numbers])
Kickstarter_Example_66()


## How to choose a random elecment from a list in Python
def Kickstarter_Example_67(): 
    print()
    print(format('How to choose a random elecment from a list in Python','*^82'))    
    import warnings
    warnings.filterwarnings("ignore")
    
    from random import choice
    # Make a list of crew members
    crew_members = ['Steve', 'Stacy', 'Miller', 'Chris', 'Bill', 'Jack']
    # Choose a random crew member
    print()
    print(choice(crew_members))
Kickstarter_Example_67()


## How to use if and if-else in Python
def Kickstarter_Example_68(): 
    print()
    print(format('How to use if and if-else in Python','*^82'))    
    import warnings
    warnings.filterwarnings("ignore")

    # declare a 'STATUS' variable
    #1 if the STATUS is active
    #0 if the STATUS is not active
    #unknown if the STATUS is unknwon
    STATUS = 1
    #If the STATUS is active print a statement
    if STATUS == 1:
        print('The STATUS is active.')

    #If the STATUS is active print a statement, if not, print a different statement
    STATUS = 0
    if STATUS == 1:
        print('The STATUS is active.')
    else: print('The STATUS is not active.')

    #If the STATUS is active print a statement, if not, print a different statement, 
    # if unknown, state a third statement.
    STATUS = 'unknown'
    if STATUS == 1:
        print('The STATUS is active.')
    elif STATUS == 'unknown':
        print('The STATUS is unknown')
    else: print('The STATUS is not active.')    
Kickstarter_Example_68()

## How to apply functions in a Group in a Pandas DataFrame
def Kickstarter_Example_69(): 
    print()
    print(format('How to apply functions in a Group in a Pandas DataFrame','*^82'))    
    import warnings
    warnings.filterwarnings("ignore")
    # load libraries
    import pandas as pd
    # Create an example dataframe
    data = {'EmployeeGroup': ['A','A','A','A','A','A','B','B','B','B','B','C','C','C','C','C'],
            'Points': [10,40,50,70,50,50,60,10,40,50,60,70,40,60,40,60]}
    df = pd.DataFrame(data)
    print('\nThe Original DataFrame'); print(df)
    # Apply A Function (Rolling Mean) To The DataFrame, By Group
    print('\nRolling Mean:'); print(df.groupby('EmployeeGroup')['Points'].apply(lambda x:x.rolling(center=False,window=2).mean()))
    # Apply A Function (Mean) To The DataFrame, By Group    
    print('\nAverage:'); print(df.groupby('EmployeeGroup')['Points'].apply(lambda x:x.mean()))    
    # Apply A Function (Sum) To The DataFrame, By Group    
    print('\nSum:'); print(df.groupby('EmployeeGroup')['Points'].apply(lambda x:x.sum()))    
    # Apply A Function (Max) To The DataFrame, By Group    
    print('\nMaximum:'); print(df.groupby('EmployeeGroup')['Points'].apply(lambda x:x.max()))    
    # Apply A Function (Min) To The DataFrame, By Group    
    print('\nMinimum:'); print(df.groupby('EmployeeGroup')['Points'].apply(lambda x:x.min()))    
Kickstarter_Example_69()

## How to do Data Analysis in a Pandas DataFrame
def Kickstarter_Example_70(): 
    print()
    print(format('How to Data Analysis in a Pandas DataFrame','*^82'))    
    import warnings
    warnings.filterwarnings("ignore")
    
    # load libraries
    import pandas as pd
    # Create dataframe
    raw_data = {'regiment': ['Nighthawks', 'Nighthawks', 'Nighthawks', 'Nighthawks', 'Dragoons', 'Dragoons', 'Dragoons', 'Dragoons', 'Scouts', 'Scouts', 'Scouts', 'Scouts'], 
                'company': ['1st', '1st', '2nd', '2nd', '1st', '1st', '2nd', '2nd','1st', '1st', '2nd', '2nd'], 
                'name': ['Miller', 'Jacobson', 'Ali', 'Milner', 'Cooze', 'Jacon', 'Ryaner', 'Sone', 'Sloan', 'Piger', 'Riani', 'Ali'], 
                'preTestScore': [4, 24, 31, 2, 3, 4, 24, 31, 2, 3, 2, 3],
                'postTestScore': [25, 94, 57, 62, 70, 25, 94, 57, 62, 70, 62, 70]}
    df = pd.DataFrame(raw_data, columns = ['regiment', 'company', 'name', 'preTestScore', 'postTestScore']); print(df)
    # Create a groupby variable that groups preTestScores by regiment
    groupby_regiment = df['preTestScore'].groupby(df['regiment'])
    # Descriptive statistics by group
    print(); print(df['preTestScore'].groupby(df['regiment']).describe())
    # Mean of each regiment’s preTestScore
    print(); print(groupby_regiment.mean())
    # Mean preTestScores grouped by regiment and company
    print(); print(df['preTestScore'].groupby([df['regiment'], df['company']]).mean())
    # Mean preTestScores grouped by regiment and company without heirarchical indexing
    print(); print(df['preTestScore'].groupby([df['regiment'], df['company']]).mean().unstack())
    # Group the entire dataframe by regiment and company
    print(); print(df.groupby(['regiment', 'company']).mean())
    # Number of observations in each regiment and company
    print(); print(df.groupby(['regiment', 'company']).size())
    # Iterate an operations over groups # Group the dataframe by regiment, and for each regiment,
    for name, group in df.groupby('regiment'): 
        # print the name of the regiment
        print(); print(name)
        # print the data of that regiment
        print(); print(group)
    # Group by columns
    print(); print(list(df.groupby(df.dtypes, axis=1)))
    print(); print(df.groupby('regiment').mean().add_prefix('mean_'))
    # Create a function to get the stats of a group
    def get_stats(group):
        return {'min': group.min(), 'max': group.max(), 'count': group.count(), 'mean': group.mean()}
    #Create bins and bin up postTestScore by those pins
    bins = [0, 25, 50, 75, 100]
    group_names = ['Low', 'Okay', 'Good', 'Great']
    df['categories'] = pd.cut(df['postTestScore'], bins, labels=group_names)
    # Apply the get_stats() function to each postTestScore bin
    print(); print(df['postTestScore'].groupby(df['categories']).apply(get_stats).unstack())
Kickstarter_Example_70()

## How to assign a new column in a Pandas DataFrame
def Kickstarter_Example_71(): 
    print()
    print(format('How to assign a new column in a Pandas DataFrame','*^82'))    
    import warnings
    warnings.filterwarnings("ignore")
    # load libraries
    import pandas as pd
    # Create empty dataframe
    df = pd.DataFrame()
    # Create a column
    df['StudentName'] = ['John', 'Steve', 'Sarah']
    # View dataframe
    print(); print(df)
    # Assign a new column to df called 'age' with a list of ages
    df = df.assign(Marks = [71, 82, 89])
    # View dataframe
    print(); print(df)    
Kickstarter_Example_71()

## How to apply arithmatic operations on a Pandas DataFrame
def Kickstarter_Example_72(): 
    print()
    print(format('How to apply arithmatic operations on a Pandas DataFrame','*^82'))    
    import warnings
    warnings.filterwarnings("ignore")
    # load libraries
    import pandas as pd
    import numpy as np
    # Create a dataframe
    data = {'name': ['Jason', 'Molly', 'Tina', 'Jake', 'Amy'], 
            'year': [2012, 2012, 2013, 2014, 2014], 
            'reports': [4, 24, 31, 2, 3],
            'coverage': [25, 94, 57, 62, 70]}
    df = pd.DataFrame(data, index = ['Cochice', 'Pima', 'Santa Cruz', 'Maricopa', 'Yuma'])
    print(); print(df)

    # Create a capitalization lambda function
    capitalizer = lambda x: x.upper()

    # Apply the capitalizer function over the column ‘name’
    # apply() can apply a function along any axis of the dataframe
    print(); print(df['name'].apply(capitalizer))

    # Map the capitalizer lambda function over each element in the series ‘name’
    # map() applies an operation over each element of a series
    print(); print(df['name'].map(capitalizer))

    # Apply a square root function to every single cell in the whole data frame
    # applymap() applies a function to every single element in the entire dataframe.
    # Drop the string variable so that applymap() can run
    df = df.drop('name', axis=1)
    print(); print(df)
    # Return the square root of every cell in the dataframe using applymap()
    print(); print(df.applymap(np.sqrt))

    # Applying A Function Over A Dataframe
    # Create a function that multiplies all non-strings by 100
    def times100(x):
        if type(x) is str: return x
        elif x:            return 100 * x
        else:              return
    
    # Apply the times100 over every cell in the dataframe
    print(); print(df.applymap(times100))
Kickstarter_Example_72()

## How to divide a list into chunks in python
def Kickstarter_Example_73(): 
    print()
    print(format('How to divide a list into chunks in python','*^82'))    
    import warnings
    warnings.filterwarnings("ignore")
    # Create a list of first names
    names = ['Steve', 'Jane', 'Sara', 'Mary','Jack','Bob', 
                   'Bily', 'Boni', 'Chris','Sori', 'Will', 'Won','Li']
    # Create a function called "chunks" with two arguments, l and n:
    def chunks(l, n):
        # For item i in a range that is a length of l,
        for i in range(0, len(l), n):
            # Create an index range for l of n items:
            yield l[i:i+n]
    # Create a list that from the results of the function chunks:
    names_ = list(chunks(names, 5))
    
    print();  print(names)
    print();  print(names_)
Kickstarter_Example_73()

## How to preprocess string data within a Pandas DataFrame
def Kickstarter_Example_74(): 
    print()
    print(format('How to preprocess string data within a Pandas DataFrame','*^82'))    
    import warnings
    warnings.filterwarnings("ignore")

    # load libraries
    import pandas as pd
    # Create a dataframe with a single column of strings
    data = {'stringData': ['Arizona 1 2014-12-23    3242.0',
                           'Iowa 1 2010-02-23       3453.7',
                           'Oregon 0 2014-06-20     2123.0',
                           'Maryland 0 2014-03-14   1123.6',
                           'Florida 1 2013-01-15    2134.0',
                           'Georgia 0 2012-07-14    2345.6']}
    df = pd.DataFrame(data, columns = ['stringData'])
    print(); print(df)

    # Search a column of strings for a pattern
    # Which rows of df['stringData'] contain 'xxxx-xx-xx'?
    print(); print(df['stringData'].str.contains('....-..-..', regex=True))

    # Extract the column of single digits
    # In the column 'stringData', extract single digit in the strings
    df['Boolean'] = df['stringData'].str.extract('(\d)', expand=True)
    print(); print(df['Boolean'])

    # Extract the column of dates
    # In the column 'raw', extract xxxx-xx-xx in the strings
    df['date'] = df['stringData'].str.extract('(....-..-..)', expand=True)
    print(); print(df['date'])

    # Extract the column of thousands
    # In the column 'stringData', extract ####.## in the strings
    df['score'] = df['stringData'].str.extract('(\d\d\d\d\.\d)', expand=True)
    print(); print(df['score'])

    # Extract the column of words
    # In the column 'stringData', extract the word in the strings
    df['state'] = df['stringData'].str.extract('([A-Z]\w{0,})', expand=True)
    print(); print(df['state'])

    # View the final dataframe
    print(); print(df)
Kickstarter_Example_74()

## How to create a dictionary from multiple lists
def Kickstarter_Example_75(): 
    print()
    print(format('How to create a dictionary from multiple lists','*^82'))    
    import warnings
    warnings.filterwarnings("ignore")
    # Create a list of names
    student_names = ['Sodoni Dogla', 'Chris Jefferson', 'Jessica Billars', 
                     'Michael Mulligan', 'Steven Johnson']
    # Create a list of marks obtained
    student_marks = [85, 46, 96, 74, 68]

    # Create a dictionary that is the zip of the two lists
    dictionary = dict(zip(student_names, student_marks))
    print(); print(dictionary)
Kickstarter_Example_75()

## How to convert categorical variables into numerical variables in Python
def Kickstarter_Example_76(): 
    print()
    print(format('How to convert categorical variables into numerical variables in Python','*^82'))    
    import warnings
    warnings.filterwarnings("ignore")
    
    # load libraries
    import pandas as pd
    # Create a dataframe
    data = {'first_name': ['Jason', 'Molly', 'Tina', 'Jake', 'Amy'], 
                'last_name': ['Miller', 'Jacobson', 'Ali', 'Milner', 'Cooze'], 
                'gender': ['male', 'female', 'male', 'female', 'female']}
    df = pd.DataFrame(data, columns = ['first_name', 'last_name', 'gender'])
    print(); print(df)
    # Create a set of dummy variables from the gender variable
    df_gender = pd.get_dummies(df['gender'])
    # Join the dummy variables to the main dataframe
    df_new = pd.concat([df, df_gender], axis=1)
    print(); print(df_new)
    # Alterative for joining the new columns
    df_new = df.join(df_gender)
    print(); print(df_new)
Kickstarter_Example_76()

## How to convert string categorical variables into numerical variables in Python
def Kickstarter_Example_77(): 
    print()
    print(format('How to convert strings into numerical variables in Python','*^82'))    
    import warnings
    warnings.filterwarnings("ignore")
    # load libraries
    import pandas as pd
    # Create dataframe
    raw_data = {'patient': [1, 1, 1, 2, 2], 
                'obs': [1, 2, 3, 1, 2], 
                'treatment': [0, 1, 0, 1, 0],
                'score': ['strong', 'weak', 'normal', 'weak', 'strong']} 
    df = pd.DataFrame(raw_data, columns = ['patient', 'obs', 'treatment', 'score'])
    print(); print(df)
    # Create a function that converts all values of df['score'] into numbers
    def score_to_numeric(x):
        if x=='strong': return 3
        if x=='normal': return 2
        if x=='weak':   return 1
    # Apply the function to the score variable
    df['score_num'] = df['score'].apply(score_to_numeric)
    print(); print(df)
Kickstarter_Example_77()

## How to convert string categorical variables into numerical variables using Label Encoder
def Kickstarter_Example_78(): 
    print()
    print(format('How to convert strings into numerical variables using Label Encoder','*^82'))    
    import warnings
    warnings.filterwarnings("ignore")
    # load libraries
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder
    # Create dataframe
    raw_data = {'patient': [1, 1, 1, 2, 2], 
                'obs': [1, 2, 3, 1, 2], 
                'treatment': [0, 1, 0, 1, 0],
                'score': ['strong', 'weak', 'normal', 'weak', 'strong']} 
    df = pd.DataFrame(raw_data, columns = ['patient', 'obs', 'treatment', 'score'])
    print(); print(df)

    # Create a function that converts all values of df['score'] into numbers
    def dummyEncode(df):
          columnsToEncode = list(df.select_dtypes(include=['category','object']))
          le = LabelEncoder()
          for feature in columnsToEncode:
              try:
                  df[feature] = le.fit_transform(df[feature])
              except:
                  print('Error encoding '+feature)
          return df
    df = dummyEncode(df)

    print(); print(df)
Kickstarter_Example_78()

## How to convert string variables into DateTime variables in Python
def Kickstarter_Example_79(): 
    print()
    print(format('How to convert string variables into DateTime variables in Python','*^82'))    
    import warnings
    warnings.filterwarnings("ignore")
    # load libraries
    import pandas as pd
    # Create a dataset with the index being a set of names
    raw_data = {'date': ['2017-09-01T01:23:41.004053', '2017-10-15T01:21:38.004053', 
                         '2017-12-17T01:17:38.004053'],
                'score': [25, 94, 57]}
    df = pd.DataFrame(raw_data, columns = ['date', 'score'])
    print(); print(df); print(df.dtypes)
    # convert strings to DateTime
    df["date"] = pd.to_datetime(df["date"])
    print(); print(df); print(df.dtypes)
Kickstarter_Example_79()

## How to insert a new column based on condition in Python
def Kickstarter_Example_80(): 
    print()
    print(format('How to insert a new column based on condition in Python','*^82'))    
    import warnings
    warnings.filterwarnings("ignore")
    # load libraries
    import pandas as pd
    import numpy as np
    # Create an example dataframe
    raw_data = {'student_name': ['Miller', 'Jacobson', 'Bali', 'Milner', 'Cooze', 'Jacon', 'Ryaner', 'Sone', 'Sloan', 'Piger', 'Riani', 'Ali'], 
                'test_score': [76, 88, 84, 67, 53, 96, 64, 91, 77, 73, 52, np.NaN]}
    df = pd.DataFrame(raw_data, columns = ['student_name', 'test_score'])
    print(); print(df)
    # Create a function to assign letter grades
    grades = []
    for row in df['test_score']:
        if row > 95:    grades.append('A')
        elif row > 90:  grades.append('A-')
        elif row > 85:  grades.append('B')
        elif row > 80:  grades.append('B-')
        elif row > 75:  grades.append('C')
        elif row > 70:  grades.append('C-')
        elif row > 65:  grades.append('D')
        elif row > 60:  grades.append('D-')
        else:           grades.append('Failed')
    # Create a column from the list
    df['grades'] = grades
    # View the new dataframe
    print(); print(df)
Kickstarter_Example_80()

## How to create a new column based on a condition in Python
def Kickstarter_Example_81(): 
    print()
    print(format('How to create a new column based on a condition in Python','*^82'))    
    import warnings
    warnings.filterwarnings("ignore")
    # load libraries
    import pandas as pd
    import numpy as np
    # Make a dataframe
    data = {'name': ['Jason', 'Molly', 'Tina', 'Jake', 'Amy'], 
            'age': [42, 52, 63, 24, 73], 
            'preTestScore': [4, 24, 31, 2, 3],
            'postTestScore': [25, 94, 57, 62, 70]}
    df = pd.DataFrame(data, columns = ['name', 'age', 'preTestScore', 'postTestScore'])
    print(); print(df)
    # Create a new column called df.elderly where the value is yes
    df['elderly@50'] = np.where(df['age']>=50, 'yes', 'no')
    df['elderly@60'] = np.where(df['age']>=60, 'yes', 'no')
    df['elderly@70'] = np.where(df['age']>=70, 'yes', 'no')    
    # View the dataframe
    print(); print(df)
Kickstarter_Example_81()

## How to create lists from Dictionary in Python
def Kickstarter_Example_82(): 
    print()
    print(format('How to create lists from Dictionary in Python','*^82'))    
    import warnings
    warnings.filterwarnings("ignore")

    # Create a dictionary
    dict = {'name': ['Jason', 'Molly', 'Tina', 'Jake', 'Amy'], 
            'age': [42, 52, 63, 24, 73], 
            'preTestScore': [4, 24, 31, 2, 3],
            'postTestScore': [25, 94, 57, 62, 70]
            }
    print(); print(dict)
    # Create a list of keys
    print(); print(list(dict.keys()))
    # Create a list of values
    print(); print(list(dict.values()))
Kickstarter_Example_82()

## How to create crosstabs from a Dictionary in Python
def Kickstarter_Example_83(): 
    print()
    print(format('How to create crosstabs from a Dictionary in Python','*^82'))    
    import warnings
    warnings.filterwarnings("ignore")

    # load libraries
    import pandas as pd
    raw_data = {'regiment': ['Nighthawks', 'Nighthawks', 'Nighthawks', 'Nighthawks', 'Dragoons', 'Dragoons', 'Dragoons', 'Dragoons', 'Scouts', 'Scouts', 'Scouts', 'Scouts'], 
                'company': ['infantry', 'infantry', 'cavalry', 'cavalry', 'infantry', 'infantry', 'cavalry', 'cavalry','infantry', 'infantry', 'cavalry', 'cavalry'], 
                'experience': ['veteran', 'rookie', 'veteran', 'rookie', 'veteran', 'rookie', 'veteran', 'rookie','veteran', 'rookie', 'veteran', 'rookie'],
                'name': ['Miller', 'Jacobson', 'Ali', 'Milner', 'Cooze', 'Jacon', 'Ryaner', 'Sone', 'Sloan', 'Piger', 'Riani', 'Ali'], 
                'preTestScore': [4, 24, 31, 2, 3, 4, 24, 31, 2, 3, 2, 3],
                'postTestScore': [25, 94, 57, 62, 70, 25, 94, 57, 62, 70, 62, 70]}
    df = pd.DataFrame(raw_data, columns = ['regiment', 'company', 'experience', 'name', 'preTestScore', 'postTestScore'])
    print(); print(df)
    # Create a crosstab table by company and regiment
    df1 = pd.crosstab(df.regiment, df.company, margins=True)
    print(); print(df1)    
    # Create more crosstabs
    df2 = pd.crosstab([df.company, df.experience], df.regiment,  margins=True)
    print(); print(df2)
    df3 = pd.crosstab([df.company, df.experience, df.preTestScore], df.regiment,  margins=True)
    print(); print(df3)    
Kickstarter_Example_83()

## How to delete duplicates from a Pandas DataFrame
def Kickstarter_Example_84(): 
    print()
    print(format('How to delete duplicates from a Pandas DataFrame','*^82'))    
    import warnings
    warnings.filterwarnings("ignore")
    
    # load libraries
    import pandas as pd
    # Create dataframe with duplicates
    raw_data = {'first_name': ['Jason', 'Jason', 'Jason','Tina', 'Jake', 'Amy'], 
                'last_name': ['Miller', 'Miller', 'Miller','Ali', 'Milner', 'Cooze'], 
                'age': [42, 42, 1111111, 36, 24, 73], 
                'preTestScore': [4, 4, 4, 31, 2, 3],
                'postTestScore': [25, 25, 25, 57, 62, 70]}
    df = pd.DataFrame(raw_data, columns = ['first_name', 'last_name', 'age', 
                                           'preTestScore', 'postTestScore'])
    print(); print(df)
    # Identify which observations are duplicates
    print(); print(df.duplicated())
    print(); print(df.drop_duplicates(keep='first'))
    # Drop duplicates in the first name column, but take the last obs in the duplicated set
    print(); print(df.drop_duplicates(['first_name'], keep='last'))
Kickstarter_Example_84()

## How to get descriptive statistics of a Pandas DataFrame
def Kickstarter_Example_85(): 
    print()
    print(format('How to get descriptive statistics of a Pandas DataFrame','*^82'))    
    import warnings
    warnings.filterwarnings("ignore")
    # load libraries
    import pandas as pd
    #Create dataframe
    data = {'name': ['Jason', 'Molly', 'Tina', 'Jake', 'Amy'], 
            'age': [42, 52, 36, 24, 73], 
            'preTestScore': [4, 24, 31, 2, 3],
            'postTestScore': [25, 94, 57, 62, 70]}
    df = pd.DataFrame(data, columns = ['name', 'age', 'preTestScore', 'postTestScore'])
    print(); print(df)
    print(); print(df.info())
    # The sum of all the ages
    print(); print(df['age'].sum())
    # Mean preTestScore
    print(); print(df['preTestScore'].mean())
    # Cumulative sum of preTestScores, moving from the rows from the top
    print(); print(df['preTestScore'].cumsum())
    # Summary statistics on preTestScore
    print(); print(df['preTestScore'].describe())
    # Count the number of non-NA values
    print(); print(df['preTestScore'].count())
    # Minimum value of preTestScore
    print(); print(df['preTestScore'].min())
    # Maximum value of preTestScore
    print(); print(df['preTestScore'].max())
    # Median value of preTestScore
    print(); print(df['preTestScore'].median())
    # Sample variance of preTestScore values
    print(); print(df['preTestScore'].var())
    # Sample standard deviation of preTestScore values
    print(); print(df['preTestScore'].std())
    # Skewness of preTestScore values
    print(); print(df['preTestScore'].skew())
    # Kurtosis of preTestScore values
    print(); print(df['preTestScore'].kurt())
    # Correlation Matrix Of Values
    print(); print(df.corr())
    # Covariance Matrix Of Values
    print(); print(df.cov())
Kickstarter_Example_85()

## How to drop ROW and COLUMN in a Pandas DataFrame
def Kickstarter_Example_86(): 
    print()
    print(format('How to drop ROW and COLUMN in a Pandas DataFrame','*^82'))    
    import warnings
    warnings.filterwarnings("ignore")
    # load libraries
    import pandas as pd
    # Create a dataframe
    data = {'name': ['Jason', 'Molly', 'Tina', 'Jake', 'Amy'], 
            'year': [2012, 2012, 2013, 2014, 2014], 
            'reports': [4, 24, 31, 2, 3]}
    df = pd.DataFrame(data, index = ['Cochice', 'Pima', 'Santa Cruz', 'Maricopa', 'Yuma'])
    print(); print(df)
    # Drop an observation (row)
    print(); print(df.drop(['Cochice', 'Pima']))
    # Drop a variable (column) # Note: axis=1 denotes that we are referring to a column, not a row
    print(); print(df.drop('reports', axis=1))
    # Drop a row if it contains a certain value (in this case, “Tina”)
    print(); print(df[df.name != 'Tina'])
    # Drop a row by row number (in this case, row 3)
    print(); print(df.drop(df.index[2]))
    # can be extended to dropping a range
    print(); print(df.drop(df.index[[2,3]]))
    # dropping relative to the end of the DF.
    print(); print(df.drop(df.index[-2]))
    # Keep top 3
    print(); print(df[:3])
    # Drop bottom 3 
    print(); print(df[:-3])
Kickstarter_Example_86()

## How to filter a Pandas DataFrame
def Kickstarter_Example_87(): 
    print()
    print(format('How to filter a Pandas DataFrame','*^82'))    
    import warnings
    warnings.filterwarnings("ignore")
    # load libraries
    import pandas as pd
    # Create Dataframe
    data = {'name': ['Jason', 'Molly', 'Tina', 'Jake', 'Amy'], 
            'year': [2012, 2012, 2013, 2014, 2014], 
            'reports': [4, 24, 31, 2, 3],
            'coverage': [25, 94, 57, 62, 70]}
    df = pd.DataFrame(data, index = ['Cochice', 'Pima', 'Santa Cruz', 'Maricopa', 'Yuma'])
    print(); print(df)
    # View Column
    print(); print(df['name'])
    # View Two Columns
    print(); print(df[['name', 'reports']])
    # View First Two Rows
    print(); print(df[:2])
    # View Rows Where Coverage Is Greater Than 50
    print(); print(df[df['coverage'] > 50])
    # View Rows Where Coverage Is Greater Than 50 And Reports Less Than 4
    print(); print(df[(df['coverage']  > 50) & (df['reports'] < 4)])
Kickstarter_Example_87()

## How to find the largest value in a Pandas DataFrame
def Kickstarter_Example_88(): 
    print()
    print(format('How to find the largest value in a Pandas DataFrame','*^82'))    
    import warnings
    warnings.filterwarnings("ignore")
    # load libraries
    import pandas as pd
    # Create dataframe
    raw_data = {'first_name': ['Jason', 'Molly', 'Tina', 'Jake', 'Amy'], 
                'last_name': ['Miller', 'Jacobson', 'Ali', 'Milner', 'Cooze'], 
                'age': [42, 52, 36, 24, 73], 
                'preTestScore': [4, 24, 31, 2, 3],
                'postTestScore': [25, 94, 57, 62, 70]}
    df = pd.DataFrame(raw_data, columns = ['first_name', 'last_name', 'age', 
                                           'preTestScore', 'postTestScore'])
    print(); print(df)
    # Index of the row with the highest and  lowest value in the preTestScore column
    print()
    print("Index of highest value: "); print(df['preTestScore'].idxmax())
    print("Index of lowest value: "); print(df['preTestScore'].idxmin())    
Kickstarter_Example_88()

## How to group rows in a Pandas DataFrame
def Kickstarter_Example_89(): 
    print()
    print(format('How to group rows in a Pandas DataFrame','*^82'))    
    import warnings
    warnings.filterwarnings("ignore")
    # load libraries
    import pandas as pd
    # Create a dataframe
    raw_data = {'regiment': ['Nighthawks', 'Nighthawks', 'Nighthawks', 'Nighthawks', 'Dragoons', 'Dragoons', 'Dragoons', 'Dragoons', 'Scouts', 'Scouts', 'Scouts', 'Scouts'], 
                'company': ['1st', '1st', '2nd', '2nd', '1st', '1st', '2nd', '2nd','1st', '1st', '2nd', '2nd'], 
                'name': ['Miller', 'Jacobson', 'Ali', 'Milner', 'Cooze', 'Jacon', 'Ryaner', 'Sone', 'Sloan', 'Piger', 'Riani', 'Ali'], 
                'preTestScore': [4, 24, 31, 2, 3, 4, 24, 31, 2, 3, 2, 3],
                'postTestScore': [25, 94, 57, 62, 70, 25, 94, 57, 62, 70, 62, 70]}
    df = pd.DataFrame(raw_data, columns = ['regiment', 'company', 'name', 'preTestScore', 'postTestScore'])
    print(); print(df)
    # Create a grouping object. In other words, create an object that
    # represents that particular grouping. 
    regiment_preScore = df['preTestScore'].groupby(df['regiment'])
    # Display the values of the each regiment's pre-test score
    print(); print(regiment_preScore.mean())
    print(); print(regiment_preScore.sum())
    print(); print(regiment_preScore.max())
    print(); print(regiment_preScore.min())    
    print(); print(regiment_preScore.count())    
Kickstarter_Example_89()

## How to present Hierarchical Data in Pandas
def Kickstarter_Example_90(): 
    print()
    print(format('How to present Hierarchical Data in Pandas','*^82'))    
    import warnings
    warnings.filterwarnings("ignore")
    # load libraries
    import pandas as pd
    # Create dataframe
    raw_data = {'regiment': ['Nighthawks', 'Nighthawks', 'Nighthawks', 'Nighthawks', 
                             'Dragoons', 'Dragoons', 'Dragoons', 'Dragoons', 'Scouts', 
                             'Scouts', 'Scouts', 'Scouts'], 
                'company': ['1st', '1st', '2nd', '2nd', '1st', '1st', '2nd', 
                            '2nd','1st', '1st', '2nd', '2nd'], 
                'name': ['Miller', 'Jacobson', 'Bali', 'Milner', 'Cooze', 'Jacon', 
                         'Ryaner', 'Sone', 'Sloan', 'Piger', 'Riani', 'Ali'], 
                'preTestScore': [4, 24, 31, 2, 3, 4, 24, 31, 2, 3, 2, 3],
                'postTestScore': [25, 94, 57, 62, 70, 25, 94, 57, 62, 70, 62, 70]}
    df = pd.DataFrame(raw_data, columns = ['regiment', 'company', 'name', 
                                           'preTestScore', 'postTestScore'])
    print(); print(df)
    # Set the hierarchical index but leave the columns inplace
    df.set_index(['regiment', 'company'], drop=False)
    print(); print(df)
    # Set the hierarchical index to be by regiment, and then by company
    df = df.set_index(['regiment', 'company'])
    print(); print(df)
    # View the index
    print(); print(df.index)
    # Swap the levels in the index
    print(); print(df.swaplevel('regiment', 'company'))
    # Summarize the results by regiment
    print(); print(df.sum(level='regiment'))
    print(); print(df.count(level='regiment'))
    print(); print(df.mean(level='regiment'))
    print(); print(df.max(level='regiment'))
    print(); print(df.min(level='regiment'))
Kickstarter_Example_90()

## How to JOIN and MERGE Pandas DataFrame
def Kickstarter_Example_91(): 
    print()
    print(format('How to JOIN and MERGE Pandas DataFrame','*^82'))    
    import warnings
    warnings.filterwarnings("ignore")
    # load libraries
    import pandas as pd
    # Create a dataframe
    raw_data = {'subject_id': ['1', '2', '3', '4', '5'],
                'first_name': ['Alex', 'Amy', 'Allen', 'Alice', 'Ayoung'], 
                'last_name': ['Anderson', 'Ackerman', 'Ali', 'Aoni', 'Atiches']}
    df_a = pd.DataFrame(raw_data, columns = ['subject_id', 'first_name', 'last_name'])
    print(); print(df_a)
    # Create a second dataframe
    raw_data = {'subject_id': ['4', '5', '6', '7', '8'],
                'first_name': ['Billy', 'Brian', 'Bran', 'Bryce', 'Betty'], 
                'last_name': ['Bonder', 'Black', 'Balwner', 'Brice', 'Btisan']}
    df_b = pd.DataFrame(raw_data, columns = ['subject_id', 'first_name', 'last_name'])
    print(); print(df_b)
    # Create a third dataframe
    raw_data = {'subject_id': ['1', '2', '3', '4', '5', '7', '8', '9', '10', '11'],
                'test_id': [51, 15, 15, 61, 16, 14, 15, 1, 61, 16]}
    df_n = pd.DataFrame(raw_data, columns = ['subject_id','test_id'])
    print(); print(df_n)
    # Join the two dataframes along rows
    df_new = pd.concat([df_a, df_b])
    print(); print(df_new)
    # Join the two dataframes along columns
    df = pd.concat([df_a, df_b], axis=1)
    print(); print(df)
    # Merge two dataframes along the subject_id value
    df = pd.merge(df_new, df_n, on='subject_id')
    print(); print(df)
    # Merge two dataframes with both the left and right dataframes using the subject_id key
    df = pd.merge(df_new, df_n, left_on='subject_id', right_on='subject_id')
    print(); print(df)
    # Merge with outer join
    df = pd.merge(df_a, df_b, on='subject_id', how='outer')
    print(); print(df)
    # Merge with inner join
    df = pd.merge(df_a, df_b, on='subject_id', how='inner')
    print(); print(df)
    # Merge with right join
    df = pd.merge(df_a, df_b, on='subject_id', how='right')
    print(); print(df)
    # Merge with left join
    df = pd.merge(df_a, df_b, on='subject_id', how='left')
    print(); print(df)
    # Merge while adding a suffix to duplicate column names
    df = pd.merge(df_a, df_b, on='subject_id', how='left', suffixes=('_left', '_right'))
    print(); print(df)
    # Merge based on indexes
    df = pd.merge(df_a, df_b, right_index=True, left_index=True)
    print(); print(df)
Kickstarter_Example_91()

## How to list unique values in a Pandas DataFrame
def Kickstarter_Example_92(): 
    print()
    print(format('How to list unique values in a Pandas DataFrame','*^82'))    
    import warnings
    warnings.filterwarnings("ignore")
    # load libraries
    import pandas as pd
    # Set ipython's max row display
    pd.set_option('display.max_row', 1000)
    # Set iPython's max column width to 50
    pd.set_option('display.max_columns', 50)

    # Create an example dataframe
    data = {'name': ['Jason', 'Molly', 'Tina', 'Tina', 'Amy'], 
            'year': [2012, 2012, 2013, 2014, 2014], 
            'reports': [4, 24, 31, 2, 3]}
    df = pd.DataFrame(data, index = ['Cochice', 'Pima', 'Santa Cruz', 
                                     'Maricopa', 'Yuma'])
    print(); print(df)
    # List unique values in the df['name'] column
    print(); print(df.name.unique())
Kickstarter_Example_92()

## How to map values in a Pandas DataFrame
def Kickstarter_Example_93(): 
    print()
    print(format('How to map values in a Pandas DataFrame','*^82'))    
    import warnings
    warnings.filterwarnings("ignore")
    # load libraries
    import pandas as pd
    # Create dataframe
    raw_data = {'first_name': ['Jason', 'Molly', 'Tina', 'Jake', 'Amy'], 
                'last_name': ['Miller', 'Jacobson', 'Ali', 'Milner', 'Cooze'], 
                'age': [42, 52, 36, 24, 73], 
                'city': ['San Francisco', 'Baltimore', 'Miami', 'Douglas', 'Boston']}
    df = pd.DataFrame(raw_data, columns = ['first_name', 'last_name', 'age', 'city'])
    print(); print(df)
    # Create a dictionary of values
    city_to_state = {'San Francisco' : 'California', 
                     'Baltimore' : 'Maryland', 
                     'Miami' : 'Florida', 
                     'Douglas' : 'Arizona', 
                     'Boston' : 'Massachusetts'}
    print(); print(city_to_state)
    # Map the values of the city_to_state dictionary to the values in the city variable
    df['state'] = df['city'].map(city_to_state)
    print(); print(df)
Kickstarter_Example_93()

## How to deal with missing values in a Pandas DataFrame
def Kickstarter_Example_94(): 
    print()
    print(format('How to deal with missing values in a Pandas DataFrame','*^82'))    
    import warnings
    warnings.filterwarnings("ignore")
    # load libraries
    import pandas as pd
    import numpy as np
    
    # Create dataframe with missing values
    raw_data = {'first_name': ['Jason', np.nan, 'Tina', 'Jake', 'Amy'], 
                'last_name': ['Miller', np.nan, 'Ali', 'Milner', 'Cooze'], 
                'age': [42, np.nan, 36, 24, 73], 
                'sex': ['m', np.nan, 'f', 'm', 'f'], 
                'preTestScore': [4, np.nan, np.nan, 2, 3],
                'postTestScore': [25, np.nan, np.nan, 62, 70]}
    df = pd.DataFrame(raw_data, columns = ['first_name', 'last_name', 'age', 'sex', 
                                           'preTestScore', 'postTestScore'])
    print(); print(df)
    # Drop missing observations
    df_no_missing = df.dropna()
    print(); print(df_no_missing)
    # Drop rows where all cells in that row is NA
    df_cleaned = df.dropna(how='all')
    print(); print(df_cleaned)
    # Create a new column full of missing values
    df['location'] = np.nan
    print(); print(df)
    # Drop column if they only contain missing values
    print(); print(df.dropna(axis=1, how='all'))
    # Drop rows that contain less than five observations
    # This is really mostly useful for time series
    print(); print(df.dropna(thresh=5))
    # Fill in missing data with zeros
    print(); print(df.fillna(0))
    # Fill in missing in preTestScore with the mean value of preTestScore
    # inplace=True means that the changes are saved to the df right away
    df["preTestScore"].fillna(df["preTestScore"].mean(), inplace=True)
    print(); print(df)
    # Fill in missing in postTestScore with each sex’s mean value of postTestScore
    df["postTestScore"].fillna(df.groupby("sex")["postTestScore"].transform("mean"), inplace=True)
    print(); print(df)
    # Select the rows of df where age is not NaN and sex is not NaN
    print(); print(df[df['age'].notnull() & df['sex'].notnull()])
    print(); print(df[df['age'].notnull() & df['sex'].notnull()].fillna(0))
Kickstarter_Example_94()

## How to calculate MOVING AVG in a Pandas DataFrame
def Kickstarter_Example_95(): 
    print()
    print(format('How to calculate MOVING AVG in a Pandas DataFrame','*^82'))    
    import warnings
    warnings.filterwarnings("ignore")
    # load libraries
    import pandas as pd
    raw_data = {'regiment': ['Nighthawks', 'Nighthawks', 'Nighthawks', 'Nighthawks', 
                             'Dragoons', 'Dragoons', 'Dragoons', 'Dragoons', 'Scouts', 
                             'Scouts', 'Scouts', 'Scouts'], 
                'company': ['1st', '1st', '2nd', '2nd', '1st', '1st', '2nd', 
                            '2nd','1st', '1st', '2nd', '2nd'], 
                'name': ['Miller', 'Jacobson', 'Bali', 'Milner', 'Cooze', 'Jacon', 
                         'Ryaner', 'Sone', 'Sloan', 'Piger', 'Riani', 'Ali'], 
                'preTestScore': [4, 24, 31, 2, 3, 4, 24, 31, 2, 3, 2, 3],
                'postTestScore': [25, 94, 57, 62, 70, 25, 94, 57, 62, 70, 62, 70]}
    df = pd.DataFrame(raw_data, columns = ['regiment', 'company', 'name', 
                                           'preTestScore', 'postTestScore'])    
    print(); print(df)

    # Calculate Rolling Moving Average with Window of 2
    df1 = df.rolling(window=2).mean()
    print(); print(df1)
    df2 = df1.fillna(0)
    print(); print(df2)
Kickstarter_Example_95()

## How to Normalise a Pandas DataFrame Column
def Kickstarter_Example_96(): 
    print()
    print(format('How to Normalise a Pandas DataFrame Column','*^82'))    
    import warnings
    warnings.filterwarnings("ignore")
    # load libraries
    import pandas as pd
    from sklearn import preprocessing
    # Create an example dataframe with a column of unnormalized data
    data = {'score': [234,24,14,27,-74,46,73,-18,59,160]}
    df = pd.DataFrame(data)
    print(); print(df)
    # Normalize The Column
    # Create x, where x the 'scores' column's values as floats
    x = df[['score']].values.astype(float)
    print(); print(x)
    # Create a minimum and maximum processor object
    min_max_scaler = preprocessing.MinMaxScaler()
    # Create an object to transform the data to fit minmax processor
    x_scaled = min_max_scaler.fit_transform(x)
    # Run the normalizer on the dataframe
    df_normalized = pd.DataFrame(x_scaled)
    # View the dataframe
    print(); print(df_normalized)
Kickstarter_Example_96()

## How to create Pivot table using a Pandas DataFrame
def Kickstarter_Example_97(): 
    print()
    print(format('How to create Pivot table using a Pandas DataFrame','*^82'))    
    import warnings
    warnings.filterwarnings("ignore")
    # load libraries
    import pandas as pd
    # Create dataframe
    raw_data = {'regiment': ['Nighthawks', 'Nighthawks', 'Nighthawks', 'Nighthawks', 
                             'Dragoons', 'Dragoons', 'Dragoons', 'Dragoons', 
                             'Scouts', 'Scouts', 'Scouts', 'Scouts'], 
                'company': ['1st', '1st', '2nd', '2nd', '1st', '1st', '2nd', 
                            '2nd','1st', '1st', '2nd', '2nd'], 
                'TestScore': [4, 24, 31, 2, 3, 4, 24, 31, 2, 3, 2, 3]}
    df = pd.DataFrame(raw_data, columns = ['regiment', 'company', 'TestScore'])
    print(); print(df)
    # Create a pivot table of group means, by company and regiment
    df1 = pd.pivot_table(df, index=['regiment','company'], aggfunc='mean')
    print(); print(df1)
    # Create a pivot table of group score counts, by company and regimensts
    df2 = df.pivot_table(index=['regiment','company'], aggfunc='count')
    print(); print(df2)
    # Create a pivot table of group score max, by company and regimensts
    df3 = df.pivot_table(index=['regiment','company'], aggfunc='max')
    print(); print(df3)
    # Create a pivot table of group score min, by company and regimensts
    df4 = df.pivot_table(index=['regiment','company'], aggfunc='min')
    print(); print(df4)    
Kickstarter_Example_97()

## How to format string in a Pandas DataFrame Column
def Kickstarter_Example_98(): 
    print()
    print(format('How to format string in a Pandas DataFrame Column','*^82'))    
    import warnings
    warnings.filterwarnings("ignore")
    # load libraries
    import pandas as pd
    # Create a list of first names
    first_names = pd.Series(['Steve Murrey', 'Jane Fonda', 
                             'Sara McGully', 'Mary Jane'])
    print()
    print(first_names)
    # print the column with lower case
    print(); print(first_names.str.lower())
    # print the column with upper case
    print(); print(first_names.str.upper())
    # print the column with title case
    print(); print(first_names.str.title())
    # print the column split across spaces
    print(); print(first_names.str.split(" "))
    # print the column with capitalized case
    print(); print(first_names.str.capitalize())
Kickstarter_Example_98()

## How to randomly sample a Pandas DataFrame
def Kickstarter_Example_99(): 
    print()
    print(format('randomly sample a Pandas DataFrame','*^82'))    
    import warnings
    warnings.filterwarnings("ignore")
    # load libraries
    import pandas as pd
    import numpy as np
    # Create dataframe
    raw_data = {'first_name': ['Jason', 'Molly', 'Tina', 'Jake', 'Amy'], 
                'last_name': ['Miller', 'Jacobson', 'Ali', 'Milner', 'Cooze'], 
                'age': [42, 52, 36, 24, 73], 
                'preTestScore': [4, 24, 31, 2, 3],
                'postTestScore': [25, 94, 57, 62, 70]}
    df = pd.DataFrame(raw_data, columns = ['first_name', 'last_name', 'age', 
                                           'preTestScore', 'postTestScore'])
    print(); print(df)
    # Select a random subset of 2 without replacement
    print(); print(df.take(np.random.permutation(len(df))[:2]))
    # Select a random subset of 4 without replacement
    print(); print(df.take(np.random.permutation(len(df))[:4]))
    # random sample of df    
    df1 = df.sample(3)
    print(); print(df1)
Kickstarter_Example_99()

## How to rank a Pandas DataFrame
def Kickstarter_Example_100(): 
    print()
    print(format('How to rank a Pandas DataFrame','*^82'))    
    import warnings
    warnings.filterwarnings("ignore")
    # load libraries
    import pandas as pd
    # Create dataframe
    data = {'name': ['Jason', 'Molly', 'Tina', 'Jake', 'Amy'], 
            'year': [2012, 2012, 2013, 2014, 2014], 
            'reports': [4, 24, 31, 2, 3],
            'coverage': [25, 94, 57, 62, 70]}
    df = pd.DataFrame(data, index = ['Cochice', 'Pima', 'Santa Cruz', 'Maricopa', 'Yuma'])
    print(); print(df)
    # Create a new column that is the rank of the value of coverage in ascending order
    df['coverageRanked'] = df['coverage'].rank(ascending=True)
    print(); print(df)
    # Create a new column that is the rank of the value of coverage in descending order
    df['coverageRanked'] = df['coverage'].rank(ascending=False)
    print(); print(df)    
Kickstarter_Example_100()

## How to reindex Pandas Series and DataFrames
def Kickstarter_Example_101(): 
    print()
    print(format('How to reindex Pandas Series and DataFrame','*^82'))    
    import warnings
    warnings.filterwarnings("ignore")
    # load libraries
    import pandas as pd
    # Create a pandas series of the risk of fire in Southern Arizona
    brushFireRisk = pd.Series([34, 23, 12, 23], 
                              index = ['Bisbee', 'Douglas', 
                                       'Sierra Vista', 'Tombstone'])
    print(); print(brushFireRisk)
    # Reindex the series and create a new series variable
    brushFireRiskReindexed = brushFireRisk.reindex(['Tombstone', 'Douglas', 
                             'Bisbee', 'Sierra Vista', 'Barley', 'Tucson'])
    print(); print(brushFireRiskReindexed)
    # Reindex the series and fill in any missing indexes as 0
    brushFireRiskReindexed = brushFireRisk.reindex(['Tombstone', 'Douglas', 
                            'Bisbee', 'Sierra Vista', 'Barley', 'Tucson'], 
                            fill_value = 0)
    print(); print(brushFireRiskReindexed)    
    # Create a dataframe
    data = {'county': ['Cochice', 'Pima', 'Santa Cruz', 'Maricopa', 'Yuma'], 
            'year': [2012, 2012, 2013, 2014, 2014], 
            'reports': [4, 24, 31, 2, 3]}
    df = pd.DataFrame(data)
    print(); print(df)
    # Change the order (the index) of the rows
    print(); print(df.reindex([4, 3, 2, 1, 0]))
    # Change the order (the index) of the columns
    columnsTitles = ['year', 'reports', 'county']
    print(); print(df.reindex(columns=columnsTitles))
Kickstarter_Example_101()

    

