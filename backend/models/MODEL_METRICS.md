# Crop Recommendation Model Metrics

## Model: Random Forest Classifier (Tuned)

### Best Hyperparameters
- n_estimators: 200
- max_depth: None
- min_samples_split: 2
- min_samples_leaf: 1

### Performance on Test Set (20% holdout)
| Metric    | Score  |
|-----------|--------|
| Accuracy  | 0.9955 |
| Precision | 0.9955 |
| Recall    | 0.9955 |
| F1-Score  | 0.9955 |

### Cross-Validation (5-fold)
| Metric         | Score  |
|----------------|--------|
| CV F1-Score    | 0.9940 |

### Model Comparison
| Model               | Accuracy | F1-Score |
|---------------------|----------|----------|
| Random Forest       | 0.9955   | 0.9955   |
| XGBoost             | 0.9932   | 0.9932   |
| Decision Tree       | 0.9773   | 0.9770   |
| Logistic Regression | 0.9659   | 0.9660   |

### Dataset Info
- Total samples: 2200
- Training samples: 1760
- Testing samples: 440
- Number of classes: 22
- Balance: Perfectly balanced (100 samples per class)
- Features: 7 (N, P, K, temperature, humidity, ph, rainfall)