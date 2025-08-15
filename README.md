# Snopes Fake-News Classifier
## Code:
1) Scraped Snopes Website for Articles from January-July 2025. 
2) Predict whether a Snopes fact-check article is True or False using classic NLP (TF-IDF) and a linear classifier.
Inputs are kept separate (title, category, body) and vectorized column-wise to preserve signal per field.

## Quick Run-down
🧹 Clean, modular text pipeline (stopwords, basic cleaning)

🔤 Column-wise TF-IDF (title / category / body) via ColumnTransformer

⚖️ Class-imbalance aware training (class_weight='balanced')

📈 Clear evaluation: classification report, confusion matrix, ROC-AUC

🎚️ Threshold tuning to trade precision/recall for the “False (fake)” class

## Data

Each row corresponds to a Snopes fact-check.

### Features

title (str)

category (str) — e.g., Politics, History, etc.

body (str) — article text (or summary/snippet)

### Target

binary_Tf (bool) — True if the claim was rated true, False otherwise

## Results
**Accuracy:** 0.9392

| Class         | Precision | Recall | F1-Score | Support |
|--------------|-----------|--------|----------|---------|
| False        | 0.93      | 0.74   | 0.83     | 35      |
| True         | 0.94      | 0.99   | 0.96     | 146     |
| **Accuracy** | —         | —      | **0.94** | 181     |
| **Macro avg**    | 0.93      | 0.86   | 0.89     | 181     |
| **Weighted avg** | 0.94      | 0.94   | 0.94     | 181     |
***
## Next steps / Ideas

🔍 GridSearchCV on TF-IDF n-grams, max_features, and C

🧪 Calibrated probabilities (Platt / isotonic) for better thresholding

🧠 Try linear SVM (LinearSVC) or distilBERT baseline (HuggingFace)

📊 Per-column interpretability: show top weighted n-grams for each class
***

@software{snopes_classifier,
  title = {Snopes Fake-News Classifier},
  author = {Dinesh, Dyuthi},
  year = {2025},
  url = {https://github.com/dyuthiii/Snopes}
}
