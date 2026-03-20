# Classification Algorithms Compared (R vs Python)

## TL;DR

* XGBoost and Random Forest gave the best overall performance on this tabular dataset
* Neural networks didn’t outperform tree-based models without more tuning
* KNN only worked properly after scaling (otherwise pretty unreliable)
* Results were broadly consistent across R and Python, with some differences due to defaults

In practice: if this were a real use case, I’d start with a tree-based model and then tune depending on whether recall or precision matters more, rather than jumping straight to more complex models.

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

| model        | acc   | prec  | rec   |
| ------------ | ----- | ----- | ----- |
| xgboost      | ~0.92 | ~0.85 | ~0.93 |
| mlp          | ~0.91 | ~0.86 | ~0.88 |
| keras        | ~0.91 | ~0.86 | ~0.88 |
| rf           | ~0.89 | ~0.84 | ~0.86 |
| knn (scaled) | ~0.89 | ~0.83 | ~0.88 |
| logistic     | ~0.87 | ~0.83 | ~0.81 |

Exact numbers vary a bit depending on tuning / randomness.

---

## Repo structure

```text
.
├── python/
│   └── classification_models.py
├── r/
│   └── classification_models.R
├── data/
│   └── Social_Network_Ads.csv
```

---

## How to run

### Python

```bash
conda activate classification
python classification_models.py
```

### R

Open the script in RStudio and run it.

---

## Takeaway

Main thing I got from this:

* For small tabular problems, tree-based models are really strong baselines
* Neural nets aren’t automatically better (especially without tuning)
* Results can shift a bit depending on library defaults, even for the same model

---

## If I kept going

* do more systematic hyperparameter tuning
* use cross-validation everywhere instead of a single split
* try a bigger dataset to see if patterns hold
