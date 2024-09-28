# Classification of Imbalanced Data Using SMOTE, Outlier Handling, and Feature Selection

This project focuses on solving the problem of **imbalanced datasets** by using **Synthetic Minority Over-sampling Technique (SMOTE)** to balance the classes, along with outlier handling and feature selection techniques to improve model performance. The project explores multiple classifiers, including Decision Tree, Naive Bayes, and Logistic Regression, to evaluate the effect of these techniques on classification metrics.

## Project Overview

The key components of this project are:
- **Outlier Detection and Handling**: Detecting outliers using boxplots and addressing them using techniques like robust scaling.
- **SMOTE for Class Imbalance**: Addressing class imbalance using SMOTE, ensuring that the minority class is better represented in the dataset.
- **Feature Selection**: Utilizing techniques such as **Mutual Information** and **Variance Threshold** to identify the most informative features for the classification task.
- **Classification Models**: Building and evaluating models like **Decision Tree**, **Naive Bayes**, and **Logistic Regression** to measure performance before and after applying data preprocessing steps.

## Key Techniques and Methods

### 1. Outlier Handling
Boxplots were generated to detect outliers in various columns, including `fnlwgt`, `capital-gain`, `capital-loss`, and `hours-per-week`. Outliers were then managed through robust scaling techniques to ensure that they do not negatively affect the model.

### 2. SMOTE (Synthetic Minority Over-sampling Technique)
Given the imbalanced nature of the dataset, SMOTE was applied to oversample the minority class. This technique helped to balance the class distribution, leading to better performance for models, especially in predicting the minority class.

### 3. Feature Selection
Feature selection methods like **Variance Threshold** and **Mutual Information** were employed to identify and select the most relevant features:
- **Top Features based on Mutual Information**: Relationship, Marital Status, Capital Gain, Age, Education, Occupation, and Hours-per-Week were identified as the most important features.
- These features contributed significantly to the performance improvement across the classifiers.

### 4. Classification Models and Evaluation
Three classification models were used to evaluate performance:
- **Decision Tree**: Showed a decrease in accuracy after applying SMOTE but improved the recall for the minority class.
- **Naive Bayes**: Achieved better overall performance after balancing the classes, with a noticeable increase in accuracy and recall.
- **Logistic Regression**: Displayed stable performance, with an accuracy of around 82.43% after outlier handling and class balancing.

### 5. Model Performance After SMOTE
- **Decision Tree**: Accuracy dropped to 79.05%, but recall for the minority class improved, showing better classification for the minority class.
- **Naive Bayes**: Accuracy is 82.25%, indicating better overall performance after balancing the dataset.
- **Logistic Regression**: Stable accuracy of 82.43% after applying outlier handling and SMOTE.

## Results and Insights

- **Outlier Handling**: Significantly improved model stability by reducing the impact of extreme values.
- **SMOTE**: Enhanced the ability to predict the minority class, especially with Naive Bayes and Decision Tree models. This came at a slight trade-off for overall accuracy in some cases (e.g., Decision Tree), but the recall for the minority class improved.
- **Feature Selection**: Reduced the dimensionality of the data, making models faster and more interpretable, while retaining predictive power.
  
Overall, handling outliers, balancing the dataset with SMOTE, and carefully selecting features led to notable improvements in classification metrics, particularly recall for the minority class.

## How to Run the Project

1. **Clone the repository**:
   ```bash
   git clone https://github.com/MohammadHR10/Income-Classification.git
   ```

2. **Install dependencies**:
   Ensure you have the required libraries installed:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Jupyter Notebook**:
   You can open and execute the notebook (`Income Classification.ipynb`) to replicate the analysis:
   ```bash
   jupyter notebook Income Income Classification.ipynb
   ```

4. **Explore the Results**:
   The notebook contains the full code for preprocessing, training, and evaluation of the models, including visualizations such as confusion matrices and classification reports.

## Dependencies

The project uses the following key Python libraries:
- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `imbalanced-learn` (for SMOTE)

## Future Improvements

- **Hyperparameter Tuning**: Applying techniques like GridSearchCV to further fine-tune the models.
- **Additional Classifiers**: Exploring more complex models like Random Forests or Support Vector Machines (SVM).
- **Ensemble Methods**: Implementing ensemble learning techniques to further improve classification performance.

Feel free to customize and modify this README based on your specific project structure and goals!
