# Food Classification with Machine Learning

A multi-class classifier that predicts food types (Pizza, Sushi, Shawarma) from student survey responses using machine learning techniques.

## Project Overview

This project analyzes survey data to classify food items based on various characteristics including complexity, ingredients, serving settings, price, movie associations, drink pairings, personal associations, and spice preferences. The final model achieves **85.1% accuracy** on the test set.

### Key Features
- **Multi-class classification** for 3 food types: Pizza, Sushi, Shawarma
- **Advanced feature engineering** from free-text and categorical survey responses
- **Comprehensive model evaluation** across multiple ML algorithms
- **Hyperparameter optimization** using Optuna
- **Robust preprocessing pipeline** handling various data types

## Technical Stack

- **Python** - Core programming language
- **Scikit-Learn** - Machine learning models and preprocessing
- **Optuna** - Hyperparameter optimization
- **NumPy/Pandas** - Data manipulation and analysis
- **Matplotlib/Seaborn** - Data visualization

## Dataset

The dataset consists of 8 survey questions covering:

1. **Complexity** (1-5 scale) - Food preparation complexity
2. **Ingredients** (free-text) - Expected number of ingredients
3. **Setting** (multi-select) - Where food is typically served
4. **Price** (free-text) - Expected cost per serving
5. **Movie Association** (free-text) - Movies associated with the food
6. **Drink Pairing** (free-text) - Preferred beverage pairings
7. **Person Association** (multi-select) - Who the food reminds you of
8. **Hot Sauce** (single-select) - Spice level preference

## Data Preprocessing

### Feature Engineering Highlights:
- **Text-to-numeric conversion**: Extracted numerical values from free-text responses
- **One-hot encoding**: Created binary features for categorical variables
- **Movie grouping**: Consolidated responses into top 20 most frequent movies + "Other"
- **Drink categorization**: Grouped similar beverages into broader categories
- **Range handling**: Converted price ranges to average values
- **Standardization**: Applied z-score normalization to numerical features

### Preprocessing Steps:
1. Handle missing values (NaN replacement with median/mean)
2. Convert text responses to numerical representations
3. Create binary indicator variables for multi-select questions
4. Standardize numerical features
5. Stratified train/validation/test split (68%/12%/20%)

## Model Development

### Models Evaluated:
- **Logistic Regression** (baseline) - 87% validation accuracy
- **Decision Trees** - Simple, interpretable model
- **Random Forest** - Ensemble method for improved generalization
- **XGBoost** - Gradient boosting with regularization
- **Multi-Layer Perceptron (MLP)** - Neural network approach ⭐ **Selected**

### Final Model Architecture:
- **Type**: Single-layer MLP
- **Hidden units**: 120 neurons
- **Activation**: Tanh
- **Solver**: SGD
- **Learning rate**: 0.0106
- **Regularization**: L2 (α = 0.0228)

## Hyperparameter Optimization

Used **Optuna** with Tree-structured Parzen Estimator (TPE) for efficient hyperparameter search:
- **30 trials** for each model type
- **3-fold cross-validation** for robust evaluation
- **Bayesian optimization** to focus search on promising regions

## Performance Metrics

### Final Model Performance:
- **Test Accuracy**: 85.1%
- **Validation Accuracy**: Similar performance across all classes
- **Evaluation Metrics**: Precision, Recall, F1-Score for each class
- **Balanced Performance**: Consistent results across Pizza, Sushi, and Shawarma

## Usage

### Prerequisites
```bash
pip install numpy pandas scikit-learn optuna matplotlib seaborn
```

### Making Predictions
```python
from pred import predict_all

# Predict on new data
predictions = predict_all('test_data.csv')

# Print results
for pred in predictions:
    print(pred)
```

### Command Line Usage
```bash
python pred.py test_data.csv
```

## Key Insights

### Feature Importance:
- **Price** - Sushi shows highest variability and median price
- **Complexity** - Clear differentiation between food types
- **Setting** - Strong associations (Pizza→parties, Shawarma→lunch)
- **Ingredients** - Distinct patterns per food type
- **Hot Sauce** - Shawarma associated with higher spice preferences

### Model Selection Rationale:
The MLP was selected over other models due to:
- Superior handling of non-linear feature interactions
- Balanced performance across all classes
- Robust generalization to unseen data
- Flexibility in capturing complex patterns

## Future Improvements

- **Ensemble Methods**: Combine multiple models for better performance
- **Feature Selection**: Identify most informative features to reduce dimensionality
- **Deep Learning**: Explore more complex neural network architectures
- **Cross-validation**: Implement more sophisticated validation strategies
- **Data Augmentation**: Generate synthetic samples to improve model robustness

## License

This project was developed as part of CSC311: Introduction to Machine Learning at the University of Toronto.

## Acknowledgments

- **Prof. Alice Gao** - Course instructor
- **University of Toronto** - Academic institution
- **CSC311 Teaching Team** - Guidance and support

---

*Project completed: April 2025*