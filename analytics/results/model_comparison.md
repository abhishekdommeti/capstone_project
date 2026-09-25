# Model comparison (test set)

| Model | Classification (survived): Accuracy | Classification (survived): Precision | Classification (survived): Recall | Classification (survived): F1 | Classification (survived): AUC | Regression (fare): MAE | Regression (fare): RMSE | Regression (fare): R² | Regression (fare): Adj. R² |
|---|---|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.809 | 0.783 | 0.691 | 0.734 | 0.861 |  |  |  |  |
| Decision Tree | 0.798 | 0.776 | 0.662 | 0.714 | 0.851 |  |  |  |  |
| Random Forest | 0.803 | 0.762 | 0.706 | 0.733 | 0.824 |  |  |  |  |
| Random Forest (tuned) | 0.831 | 0.828 | 0.706 | 0.762 | 0.841 |  |  |  |  |
| Linear Regression (fare) |  |  |  |  |  | 21.140 | 41.747 | 0.347 | 0.312 |

_Classification and regression metrics are on different scales and are not directly comparable; they are shown as two separate metric groups. All values are on the held-out test set._

## Final recommendation

I would deploy **Random Forest (tuned)** (ranked by test F1, with AUC as tie-breaker): F1 0.762, AUC 0.841, accuracy 0.831, precision 0.828, recall 0.706. The runner-up, Logistic Regression, scores F1 0.734 / AUC 0.861, a margin of 0.028 F1. As a tuned ensemble it is more stable than a single tree and its OOB score gives a second, independent estimate, at the cost of being harder to explain. For reference, the Logistic Regression baseline scores F1 0.734 / AUC 0.861, so any extra complexity should be judged against that. Caveat: this is one 20% split (178 test rows), and the tuned forest's OOB score is 0.833; the regression model is a separate task and is not ranked against the classifiers.
