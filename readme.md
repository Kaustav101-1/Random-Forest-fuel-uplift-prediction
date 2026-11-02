# Aviation Fuel Uplift Prediction

Predicting how much jet fuel (in litres) an aircraft uplifts at departure, using the route, the aircraft type and the landed fuel price at both ends of the sector.

The project compares a **linear regression** with a **random forest**, measured against a simple "predict the average" benchmark.

> **Note on data:** the dataset is **simulated** to mirror the structure of real flight-wise uplift records. The results show that the pipeline works end to end. They are not a claim about real-world accuracy, since real uplift data will be noisier.

---

## Why this problem

Airlines don't refuel to the same level on every flight. How much they uplift depends on the sector, the aircraft and, potentially, fuel price differences between airports. Buying extra fuel where it is cheaper is known as *tankering*.

A good uplift forecast helps fuel suppliers and airport operators plan stock, hydrant capacity and supply logistics.

**Uplift is not the same as fuel burn.** Uplift also depends on fuel already on board and on commercial decisions, so the two quantities are not treated as interchangeable here.

---

## Data

Three separate files, split chronologically so the model is always tested on a later period than it learned from:

| File | Flights | Purpose |
|---|---|---|
| `training.xlsx` | 15,385 | Fit the models |
| `validation.xlsx` | 3,701 | Compare models and tune the random forest |
| `test.xlsx` | — | Held back for one final evaluation |

**Inputs**

| Column | Type |
|---|---|
| Day of Week | Categorical |
| Aircraft | Categorical |
| Turboprop | Categorical (Yes/No) |
| To IATA (destination) | Categorical |
| From Landed Price (Rs/KL) | Numeric |
| To Landed Price (Rs/KL) | Numeric |

**Target:** `Qty Uplifted (L)`

---

## Method

Both models use a scikit-learn `Pipeline`, so all preprocessing is learned from the training data only and applied identically at prediction time.

- **Cleaning:** consistent text formatting, numeric parsing, and validation of weekday and Yes/No fields. Invalid prices become missing values.
- **Missing values:** the median for prices and an `"Unknown"` category for text fields, both fitted on training data.
- **Encoding:** one-hot encoding. Categories never seen in training are ignored and flagged in the console.
- **Linear regression:** inputs scaled, with one dummy column dropped per category.
- **Random forest:** 500 trees with a fixed random seed. A 12-combination grid of `min_samples_leaf` and `max_features` is scored on validation MAE.
- **Feature importance:** permutation importance on validation data. Each input is shuffled in turn to measure how much the error worsens, which gives one score per original column.

---

## Results (validation set)

| Model | MAE (L) | RMSE (L) | R² |
|---|---|---|---|
| Benchmark (training mean) | 2,997 | — | — |
| Linear regression | 473 | 582 | 0.966 |
| **Random forest** | **353** | **469** | **0.978** |

The random forest cuts average error by about **25%** compared with linear regression.

That isn't a tuning fluke: every forest setting except the `sqrt` ones scored 363–383 L MAE, all better than the linear model.

**Why the forest does better:** linear regression adds each effect independently. In reality, the effect of a route depends on the aircraft flying it, and a forest picks up those interactions automatically.

**Best settings:** `min_samples_leaf = 5`, `max_features = 0.5`

### What drives the prediction

Increase in MAE when each input is shuffled:

| Input | MAE increase (L) |
|---|---|
| Destination (To IATA) | 1,712 |
| Turboprop | 808 |
| Aircraft | 629 |
| Destination landed price | 89 |
| Origin landed price | 24 |
| Day of week | 5 |

- **Route and aircraft explain most of the uplift.** The destination is effectively standing in for sector distance.
- **Aircraft and Turboprop overlap.** Turboprop follows from aircraft type, so read the two together as one aircraft effect.
- **Price differences add relatively little** in this data, which points to a weak tankering signal. This is an association found by a predictive model, not evidence that prices *cause* changes in uplift.

> Test-set results will be added after a single final run of both models.

---

## How to run

```bash
pip install -r requirements.txt

# 1. Linear regression (writes outputs/)
python fuel_uplift_linear.py

# 2. Random forest (writes outputs_rf/ and compares against the linear results)
python fuel_uplift_rf.py

# Final, one-time evaluation on the held-out test set
python fuel_uplift_linear.py --evaluate-test
python fuel_uplift_rf.py --evaluate-test
```

Keep the three Excel files in the same folder as the scripts.

**Outputs**

| File | Contents |
|---|---|
| `*_predictions*.xlsx` | Every flight with its predicted uplift and error |
| `evaluation_metrics*.csv` | MAE, RMSE and R² per dataset |
| `tuning_results.csv` | Validation MAE for each forest setting |
| `feature_importance_rf.csv` | Permutation importance |
| `linear_vs_rf.csv` | Side-by-side comparison |
| `*.joblib` | Saved model and preprocessing, ready to load and predict |

### Predicting a new flight

```python
import joblib
import pandas as pd

model = joblib.load("outputs_rf/rf_fuel_uplift_model.joblib")

flight = pd.DataFrame([{
    "Day of Week": "Monday",
    "Aircraft": "320",
    "Turboprop": "No",
    "To IATA": "DEL",
    "From Landed Price (Rs/KL)": 91974,   # illustrative values
    "To Landed Price (Rs/KL)": 85358,
}])

print(model.predict(flight))   # ≈ 9,078 L on the simulated data
```

---

## Limitations and next steps

- **Simulated data.** Real records would add noise from weather, payload and operational decisions.
- **Missing inputs.** Fuel on board before refuelling, payload and sector distance aren't included. Any of these could improve accuracy, provided they are known at prediction time.
- **Unfamiliar routes or aircraft.** Predictions for categories absent from training are unreliable, and the scripts flag them.
- **Planned additions:** a final test-set evaluation, actual-vs-predicted and residual plots, error broken down by sector and aircraft, and time-aware cross-validation for tuning.

---

## Tech stack

Python · pandas · NumPy · scikit-learn · joblib · openpyxl

## Author

**Kaustav Palit**. EPGP, IIM Bangalore, with about 9 years in aviation fuel operations at IOCL and BPCL.
