# Data Dictionary: Travel Insurance Prediction

**Source:** Tour & Travels Company, India. Customer sales data from the 2019 travel insurance package offering.
**File:** `data/raw/TravelInsurancePrediction.csv`
**Rows:** 1,987 customers (no missing values, no duplicates)
**Task:** Binary classification: predict whether a customer will purchase the package.

---

## Variable Definitions

| Column | Type | Values / Range | Description |
|---|---|---|---|
| `Age` | Integer | 25–35 (approx.) | Customer age in years |
| `Employment Type` | Categorical | `Government Sector`, `Private Sector/Self Employed` | Employment sector |
| `GraduateOrNot` | Binary string | `Yes`, `No` | Whether the customer holds a college degree |
| `AnnualIncome` | Integer | ~300,000–1,800,000 INR | Annual income in Indian Rupees, **rounded to nearest Rs 50,000** |
| `FamilyMembers` | Integer | 2–9 | Total number of family members |
| `ChronicDiseases` | Binary int | `0`, `1` | 1 = customer has a chronic condition (diabetes, hypertension, asthma, etc.) |
| `FrequentFlyer` | Binary string | `Yes`, `No` | Derived label: `Yes` if customer booked >= 4 flights in 2017–2019 |
| `EverTravelledAbroad` | Binary string | `Yes`, `No` | Whether the customer has ever travelled internationally |
| `TravelInsurance` | Binary int | `0`, `1` | **Target**: 1 = purchased the 2019 travel insurance package |

---

## Target Distribution

| Class | Count | % |
|---|---|---|
| 0: Did not purchase | ~1,277 | ~64% |
| 1: Purchased | ~710 | ~36% |

Moderate class imbalance (~64/36). Accuracy alone is an unreliable metric.
Primary evaluation metric: **ROC-AUC**.

---

## Known Data Quality Notes

- `AnnualIncome` is rounded to the nearest Rs 50,000. This introduces discretisation error near breakpoints but is not a data quality problem; it reflects how income was collected.
- `FrequentFlyer` is a derived binary label; the raw booking counts are not available.
- `ChronicDiseases` shows near-zero correlation with purchase despite the intuition that health risk drives insurance uptake. This is a genuine empirical finding, not a data error.
- All data is from 2019 (pre-COVID). Customer behaviour and travel norms may have shifted substantially post-pandemic.

---

## Feature Engineering Applied

See `travel_insurance/features.py` for the full preprocessing pipeline.

| Group | Features | Transformation |
|---|---|---|
| Numerical (scaled) | `Age`, `AnnualIncome`, `FamilyMembers` | `StandardScaler` |
| Binary integer (passthrough) | `ChronicDiseases` | No transform |
| Binary string | `GraduateOrNot`, `FrequentFlyer`, `EverTravelledAbroad` | `OrdinalEncoder` (No=0, Yes=1) |
| Nominal | `Employment Type` | `OneHotEncoder` (drop first) |
