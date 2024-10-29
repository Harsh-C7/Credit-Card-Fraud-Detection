## Credit Card Fraud Detection

This project implements a machine learning model to detect fraudulent credit card transactions using logistic regression. The code leverages Python's [`pandas`](https://pandas.pydata.org/) for data manipulation, [`numpy`](https://numpy.org/) for numerical operations, and [`scikit-learn`](https://scikit-learn.org/stable/) for building and evaluating the machine learning model.

### Dataset
The dataset used is `creditcard.csv`, which contains transaction details and a target variable indicating whether a transaction is legitimate or fraudulent.

### Key Steps in the Code

1. **Data Loading and Exploration:**
   - The dataset is loaded using `pandas` and initial exploration is performed using `head()` to preview the data.
   - The distribution of the target variable (`Class`) is examined using `value_counts()` to understand the imbalance between legitimate and fraudulent transactions.

2. **Data Balancing:**
   - To address class imbalance, a random sample of legitimate transactions is taken to match the number of fraudulent transactions, ensuring a balanced dataset for training.

3. **Feature and Target Separation:**
   - Features (`x`) and target (`y`) variables are separated. The target variable is the "Class" column, indicating transaction legitimacy.

4. **Data Splitting:**
   - The dataset is split into training and testing sets using `train_test_split` with stratified sampling to maintain the distribution of the target variable.

5. **Model Training:**
   - A `LogisticRegression` model is instantiated and trained on the training data (`x_train`, `y_train`) with a maximum iteration of 1000 to ensure convergence.

6. **Model Evaluation:**
   - The accuracy of the model is 94% on both the training and testing datasets, providing insights into the model's performance.

### Output
- The code outputs the accuracy of the model on both the training and testing datasets, demonstrating its effectiveness in detecting fraudulent transactions.

This project serves as a practical example of using logistic regression for binary classification tasks in financial fraud detection.        
