# Classification Algorithms Compared (R vs Python)

## TL;DR

* XGBoost and Random Forest performed best on this tabular dataset
* Neural networks did not outperform tree-based models without heavy tuning
* KNN only worked well after scaling
* Results were consistent across R and Python, with some differences due to library defaults

Main takeaway: for small tabular problems like this, tree-based models are a strong first choice, and implementation details can matter more than model choice.

---

## What this is

I wanted to get a better feel for how different classification models behave on a simple tabular dataset, and also see how results compare between R and Python.

So I trained a handful of models in both languages and compared the results side by side.

---

## Data

Using the Social Network Ads dataset.

Goal is just to predict whether someone purchases based on:

* Age
* Estimated Salary

Nothing fancy — small dataset, but good for comparing model behavior.

---

## Models I tried

* Logistic regression
* KNN
* Random forest
* XGBoost
* Neural nets (sklearn + TensorFlow/Keras)

---

## What stood out

A few things were pretty consistent:

* Tree models (XGBoost, random forest) were the strongest overall
* Logistic regression worked fine but was clearly too simple
* KNN only worked well after scaling (otherwise pretty bad)
* Neural nets didn’t really beat tree models here

---

## R vs Python

Results were mostly similar across both, which was reassuring.

A couple differences though:

* Random forest did noticeably better in R — probably different defaults
* Neural net results varied a lot depending on how they were set up
* XGBoost was very consistent between the two

So overall same story, but some implementation differences matter.

---

## Rough results (Python example)

| model   | acc   | prec  | rec   |
| ------- | ----- | ----- | ----- |
| xgboost | ~0.92 | ~0.85 | ~0.93 |
| mlp     | ~0.91 | ~0.86 | ~0.88 |
| keras   | ~0.91 | ~0.86 | ~0.88 |
