TLDR;
I compared several common classification algorithms (logistic regression, random forest, XGBoost, k-nearest neighbor, and neural networks) on a simple customer purchase dataset to see which approaches generalize best. Tree-based models and a neural network performed best, reaching about 93% accuracy, while the exercise highlights how preprocessing choices, threshold tuning, and evaluation metrics can meaningfully change how a model performs in practice.

Customer Purchase Prediction: Comparing Classification Algorithms

This project compares several widely used machine learning approaches for predicting whether a user will purchase a product based on basic demographic features.

The goal was not just to train a classifier, but to explore a practical question that often arises in applied data science:

If multiple models can solve a classification problem, how much difference does the algorithm actually make, and what role do preprocessing, threshold tuning, and evaluation metrics play in the final result?

This kind of analysis is common in product analytics and marketing contexts where teams need to predict behaviors such as:

which users are likely to purchase

which leads are worth targeting

which visitors are likely to convert

The exercise also highlights how different algorithms behave when working with a relatively simple dataset.

Dataset

The dataset contains basic customer information including:

age

salary

whether the user purchased a product (binary outcome)

The data comes from a Kaggle dataset commonly used for classification demonstrations. Even though the dataset is simple, it provides a good environment for comparing algorithms and exploring how model choices affect performance.

Exploratory analysis shows that purchase probability increases for users who are both older and have higher salaries, which provides useful signal for classification models.

Modeling approach

The dataset was split into:

70% training data
30% test data

Models were trained on the training set and evaluated on the holdout test set to measure how well they generalize to unseen data.

Performance was evaluated using three metrics:

Accuracy
Sensitivity (true positive rate)
Specificity (true negative rate)

In many real applications, these metrics represent different business priorities. For example:

Sensitivity answers: “How many potential buyers did we correctly identify?”
Specificity answers: “How many non-buyers did we avoid targeting?”

Depending on the use case, one may matter more than the other.

Algorithms evaluated

Logistic Regression

Logistic regression is a classic linear model used for binary classification. It estimates the probability that a given observation belongs to the positive class.

Two variants were tested:

Default classification threshold (0.5)
Optimized threshold based on the ROC curve

Adjusting the threshold produced a more balanced tradeoff between sensitivity and specificity.

Random Forest

Random forest is an ensemble method that combines many decision trees trained on bootstrapped samples of the data.

Each tree votes on the final classification, which typically improves generalization and reduces overfitting.

In this dataset, the random forest model performed extremely well and achieved the highest overall accuracy.

XGBoost

XGBoost is a gradient boosting algorithm that sequentially builds trees that correct the errors of previous trees.

Two tuning approaches were explored:

Optimizing the number of boosting rounds
Grid search over hyperparameters

Both approaches produced high-performing models with accuracy comparable to the random forest.

K-Nearest Neighbor

KNN classifies observations based on the labels of their nearest neighbors in feature space.

One key challenge with KNN is feature scaling. Because age and salary exist on very different numeric scales, the model performed poorly before scaling.

After scaling the features to a common range, performance improved dramatically.

Neural Networks

Two neural network approaches were tested:

An R-based neural network implementation
A TensorFlow/Keras model

The TensorFlow-based neural network achieved strong performance and reached accuracy similar to the best-performing tree-based models.

Results

Several algorithms performed similarly well on this dataset.

Random Forest: about 93% accuracy
XGBoost (tuned): about 93% accuracy
Keras Neural Network: about 93% accuracy

Logistic regression and scaled KNN also performed reasonably well, but slightly below the top-performing models.

One important takeaway is that the simplicity of the dataset likely limits how much the algorithms can differentiate themselves. With only two predictive features, many models can learn the decision boundary effectively.

Key insights

Feature scaling can have a dramatic effect on distance-based algorithms like KNN.

Decision threshold tuning can significantly change how a classifier balances different types of errors.

Ensemble methods like random forests and gradient boosting often perform very well even with minimal tuning.

For simple datasets with strong signal, many algorithms converge to similar performance levels.

Tools used

R programming language

Libraries used include:

tidyverse
caret
randomForest
xgboost
class
keras
tensorflow
