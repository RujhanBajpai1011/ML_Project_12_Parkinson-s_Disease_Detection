# 🧠 Parkinson's Disease Detection

This project develops a predictive model to detect Parkinson's Disease based on various voice measurements. It employs a Support Vector Machine (SVM) classifier, along with data preprocessing techniques like standardization, to accurately classify individuals.

## **📊 Dataset**

The dataset used is parkinsons.data. It contains a range of biomedical voice measurements from 31 people, 23 of whom have Parkinson's Disease. The status column (0 for healthy, 1 for Parkinson's) is the target variable.

Key features in the dataset include:

* **MDVP:Fo(Hz), MDVP:Fhi(Hz), MDVP:Flo(Hz)**: Average, maximum, and minimum fundamental frequency.
* **Various Jitter measures**: (MDVP:Jitter(%), MDVP:Jitter(Abs), MDVP:RAP, MDVP:PPQ, Jitter:DDP).
* **Various Shimmer measures**: (MDVP:Shimmer, MDVP:Shimmer(dB), Shimmer:APQ3, Shimmer:APQ5, MDVP:APQ, Shimmer:DDA).
* **NHR, HNR**: Noise-to-HNR ratio.
* **RPDE, DFA, PPE**: Nonlinear dynamical complexity measures.
* **spread1, spread2, D2**: Pitch perturbation parameters.

## **✨ Features**

* **Data Loading and Initial Exploration**: Loads the parkinsons.data dataset into a pandas DataFrame, displaying the first few rows, checking its shape, and getting a summary of data types.

* **Missing Value Check**: Confirms the absence of missing values, ensuring the dataset is clean.

* **Statistical Summary**: Provides descriptive statistics for numerical features, offering insights into their distribution.

* **Target Variable Distribution**: Analyzes the balance of the status target variable (healthy vs. Parkinson's patients). It also shows the mean of other features grouped by status, revealing patterns related to the disease.

* **Feature and Target Separation**: Splits the dataset into features (X, excluding 'name' and 'status') and the target variable (Y, 'status').

* **Data Splitting**: Divides the data into training and testing sets (80% training, 20% testing).

* **Data Standardization**: Applies StandardScaler to normalize the feature values. This is crucial for SVM models, which are sensitive to feature scales.

* **SVM Model Training**: Trains a Support Vector Machine (SVM) classifier with a linear kernel on the scaled training data.

* **Model Evaluation**: Calculates and prints the accuracy score of the trained model on both the training and unseen test datasets.

* **Predictive System**: Includes a practical example demonstrating how to use the trained model to predict Parkinson's disease for new, unseen voice measurement data.

## **🛠️ Technologies Used**

* **Python**

* **pandas**: For robust data loading and manipulation.

* **numpy**: For efficient numerical operations.

* **scikit-learn**: For machine learning functionalities, specifically:
  * train_test_split: For dividing data.
  * StandardScaler: For feature scaling.
  * svm.SVC: The Support Vector Classifier model.
  * accuracy_score: For evaluating model performance.

## **📦 Requirements**

To run this project, you will need the following Python libraries:

* pandas
* numpy
* scikit-learn

## **🚀 Getting Started**

Follow these steps to get a copy of this project up and running on your local machine:

### **Installation**

1. **Clone the repository (if applicable):**

```
git clone <repository_url>
cd <repository_name>
```

2. **Install the required Python packages:**

```
pip install pandas numpy scikit-learn
```

### **Usage**

1. **Place the dataset**: Ensure the parkinsons.data file is located in the same directory as the Jupyter notebook (Parkinson's_Disease_Detection.ipynb).

2. **Run the Jupyter Notebook**: Open and execute all the cells in the Parkinson's_Disease_Detection.ipynb notebook using a Jupyter environment (e.g., Jupyter Lab, Jupyter Notebook, Google Colab).

The notebook will:

* Load and prepare the Parkinson's data.
* Scale the features.
* Train the SVM model.
* Output the model's accuracy on training and test data.
* Demonstrate a prediction for a sample input, indicating whether the person has Parkinson's Disease.

## **📈 Results**

The notebook provides accuracy scores for the SVM model. Based on the provided snippets, the model achieves high performance in detecting Parkinson's disease:

* **Accuracy on Training Data**: Approximately 0.8846 (88.46%)
* **Accuracy on Test Data**: Approximately 0.8718 (87.18%)

These results indicate that the SVM model is quite effective in predicting Parkinson's Disease based on voice measurements and generalizes well to unseen data.

## **🧑‍💻 Contributing**

Contributions, issues, and feature requests are highly appreciated! Feel free to fork the repository and submit pull requests.

1. Fork the repository.
2. Create your feature branch (git checkout -b feature/your-feature-name).
3. Make your changes.
4. Commit your changes (git commit -m 'Add new feature').
5. Push to the branch (git push origin feature/your-feature-name).
6. Open a Pull Request.

## **📄 License**

This project is open-source and available under the MIT License.
