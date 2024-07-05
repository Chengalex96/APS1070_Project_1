**APS1070 - Project 1: Basic Principles and Models**

**Objective:** To analyze the breast cancer dataset located on Sklearn, can we predict cancer cells using other key characteristics?

**Summary of Project 1: Breast Cancer Dataset - KNN**
-	Determine which attributes can predict Malignant or Benign 
-	Created a Dataframe
-	Using Seaborn and matplotlib to visualize
-	Standardize data, mean and variance
-	Split the data to train and test data
-	KNN Classifier, Nearest neighbors
  -	Determine which number of k provides the best accuracy
  -	Using Sklearn built-in model to calculate KNN
  -	Too little doesn’t provide enough info, too much has overfitting
-	Feature Selection, remove the features that don’t provide better accuracy
  -	Drop one feature and calculate the accuracy, then do for all features, permanently drop the feature that doesn’t affect the accuracy, ie. Choose the feature dropped that has the highest score.
  -	Stop when it reaches a cut-off (0.95*Max)
-	Standardize the data to improve the accuracy since they would have different accuracy distances based on the scale for KNN
-	Worked very well on the test data, dropped the non-important features, standardized the data, and chose the optimal k based on training data


**Part 1: Getting Started**
- Convert the data to a Pandas Dataframe using: df = pd.DataFrame(dataset.data, columns = dataset.feature_names)
- Find the key features and target classes of the dataset
- Create a column for the target that will be separated from the data for training later on:
  - df['target'] = pd.Series(dataset.target)
- Create a mask for training
  - malignant_samples= np.sum(target_data ==0)
  - benign_samples= np.sum(target_data ==1)
Effects of Standardization (Visual)
- Using seaborn.lmplot, can plot two characteristics together as X and Y
  - Different targets are marked with different colors
  - sns.lmplot(x = 'mean compactness', y ='mean concavity', data=df, hue='target')
  - sns.lmplot(x = 'mean texture', y ='mean area', data=df, hue='target')
- Standardize Data and plot data again
  - mu,si = df.mean(), df.std() #Calculate the overall mean and standard deviation of the quality scores
  - df_standard = df - mu #Subtract the mean from every entry
  - df_standard = df_standard / si #Divide every entry by the standard deviation
  - ![image](https://github.com/Chengalex96/APS1070/assets/81919159/ae82378b-d6a9-43f6-81a1-cd8f13192677)

Splitting the Data: Training and Validation Dataset
- (Creating a copy since it will be used multiple times)
  - from sklearn.model_selection import train_test_split
  - X_train, X_test, y_train, y_test = train_test_split(x, target_data, test_size=0.3, random_state=0)

**Part 2: KNN Classifier without Standardization**
- Will use SKlearn to implement K-Nearest Neighbors(KNN) to be able to predict a target class based on the number of nearest neighbors surrounding the data point
-  Find the best k that provides the best accuracy
-  Have to avoid underfitting/overfitting
-  Cross validation takes many folds of the data
  
Train KNN Classifier with 5 neighbors using cross-validation (Example)
![image](https://github.com/Chengalex96/APS1070/assets/81919159/0444d030-95ac-4775-8982-8dca916dd045)

Train KNN Classifier using cross-validation on dataset + Plotting KNN from 1 to 100 neighbors:
![image](https://github.com/Chengalex96/APS1070/assets/81919159/ab8d51ce-587a-4441-8bee-dae547edec79)

Indicate the best K-value and its accuracy:
- print(np.argmax(crossvalacc) + 1)
- print(np.amax(crossvalacc))
  
![image](https://github.com/Chengalex96/APS1070/assets/81919159/65a8014f-1421-466f-9dc2-e1015a72bf65)

Plotting KNN from 1 to 100 neighbors using cross_validate data to show underfitting and overfitting:
![image](https://github.com/Chengalex96/APS1070/assets/81919159/77aa1d1c-b680-40cc-811e-794cbea2ee29)
![image](https://github.com/Chengalex96/APS1070/assets/81919159/65481fa2-3be3-4c80-aa6f-98d9435689ac)


**Overfitting & Underfitting**
Small numbers of K lead to overfitting (K < 7) - Training score is very high and low test score - the training data train samples, test data will be different so it will blindly follow the training model. 
Large numbers of k lead to underfitting (K > 7) - Training and Test scores start to decrease (Higher error for both) - with so many samples into consideration, it will include wrong samples into training. 

**Part 3: Feature Selection**
- Want only to include the features that can help distinguish the targets otherwise there are 2^F different cases
- This is done by running a loop and removing 1 column at a time, the column with the highest accuracy will have that column removed since it has no impact on the score
- This will take F(F+1)/2
- The column is permanently removed and the process is done until when the feature is removed will decrease the accuracy below 95% of full cross-validation accuracy

Feature Selector:
- The function returns the index for the feature that was dropped that provided us with the largest cross-validation accuracy and the accuracy value
- This means that the feature has the smallest impact on distinguishing the target classes
- From the previous graph, we could observe that as we pass around k=20, it tends to underfit
- Will limit the k to 20 to improve run time

![image](https://github.com/Chengalex96/APS1070/assets/81919159/901634be-d244-441c-8c89-83996e113bb0)

Threshold Finder: 

![image](https://github.com/Chengalex96/APS1070/assets/81919159/74a8bffd-fde3-409d-82f6-1fc1a1c057d2)

Used a while loop to call the two functions to easily visualize which columns to drop:
![image](https://github.com/Chengalex96/APS1070/assets/81919159/066662cd-3669-48ba-8cb8-f1bf34073f74)

Can indicate which features are key to distinguish the correct targets:

![image](https://github.com/Chengalex96/APS1070/assets/81919159/026d56de-3301-4bc5-a78b-33b79a419e60)

**Part 4: Standardizaton**
- Standardization is key to KNN Method since it's scaled to calculate the distance between neighbors.
Notes for standardization: We can standardize the training set, not the training set, we can normalize the test set USING training set data which may not always be a mean of 0 and a standard deviation of 1. Fit StandardScaler only on the training set, and transform both sets with that scaler

Same steps but on standardized data:
![image](https://github.com/Chengalex96/APS1070/assets/81919159/e2e96782-cf6f-42de-a988-6cb00d590d42)

![image](https://github.com/Chengalex96/APS1070/assets/81919159/b497813b-8ad1-4b14-9bfb-11402be32f59)

- Standardization helped the model and its performance by scaling all the parameters.
- Since not all scales are the same, those with larger or smaller numbers are weighted differently when measuring the 'k nearest neighbors'
- Standardization led to a higher cross-validation accuracy at every number of features removed. (model predicts the correct target more often)
- Highest cross-validation accuracy occurred after dropping 18 features (12 Features remaining) - 98.5% Cross-validation Accuracy
- Best K at 13 nearest neighbors
- Features remaining: 'mean texture', 'mean perimeter', 'mean smoothness', 'radius error', 'concavity error', 'concave points error', 'fractal dimension error', 'worst texture', 'worst perimeter', 'worst area', 'worst concavity', 'worst concave points'
- Index dropped in order: 8, 11, 13, 5, 10, 20, 5, 0, 2, 19, 3, 11, 14, 9, 3, 5, 4, 12 - Used in part 5 to recreate optimized model with test data

**Part 5: Apply on test data**
![image](https://github.com/Chengalex96/APS1070/assets/81919159/16f828d1-ced4-4a6f-9ab6-2496860f054b)



