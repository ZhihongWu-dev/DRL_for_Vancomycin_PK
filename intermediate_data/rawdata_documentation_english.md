# Raw Data Documentation

## 1. Dataset Overview

The dataset (`rawdata.xlsx`) contains **raw, event-level clinical time-series data** collected during patients’ hospital or ICU stays. It integrates **laboratory test results**, **vital signs**, and **Vancomycin-related medication information**. The data is intended for clinical analysis, risk prediction, and medication-related modeling tasks.

- **File format**: Excel (`.xlsx`)
- **Number of sheets**: 1 (`Sheet1`)
- **Dataset size**: 13,345 rows × 27 columns
- **Temporal resolution**: Irregular (event-based, not fixed intervals)

Each row corresponds to a single clinical event (e.g., a lab test, a vital sign measurement, or a drug-related record) aligned on a relative hospital time axis.

---

## 2. Identifiers and Index Structure

| Column Name | Description | Notes |
|------------|-------------|-------|
| `subject_id` | Unique patient identifier | Constant across admissions |
| `hadm_id` | Hospital admission identifier | One per admission |
| `stay_id` | ICU / hospital stay identifier | Primary unit of analysis |
| `rel_time` | Relative time (hours) | Hours since `intime` |

**Recommended composite key**:
`subject_id + hadm_id + stay_id + rel_time`

---

## 3. Event Type Definition

| Column Name | Description |
|------------|-------------|
| `event_type` | Type of clinical event |
| `itemid` | Item code of the event | May be null |

The `event_type` field distinguishes different data sources, mainly including:

- `lab` – laboratory measurements
- `vitals` – vital sign measurements

Multiple events of different types may occur at the same relative time.

---

## 4. Vancomycin-Related Variables

The following variables are populated only during Vancomycin treatment periods; otherwise, they are missing (NaN).

| Column Name | Description | Unit / Notes |
|------------|-------------|--------------|
| `totalamount_mg` | Total administered dose | mg |
| `starttime` | Drug administration start time | Absolute timestamp |
| `vanco_start_rel` | Relative start time | Hours since `intime` |
| `vanco_end_rel` | Relative end time | Hours since `intime` |
| `vanco_level` | Vancomycin blood concentration | Typically trough level |

---

## 5. Laboratory Measurements (Labs)

These variables are mainly recorded when `event_type = 'lab'`.

| Column Name | Description | Clinical Meaning |
|------------|-------------|------------------|
| `creatinine` | Serum creatinine | Kidney function indicator |
| `wbc` | White blood cell count | Infection / inflammation |
| `bun` | Blood urea nitrogen | Renal function |
| `charttime` | Lab measurement time | Absolute timestamp |

---

## 6. Vital Signs (Vitals)

These variables are typically recorded when `event_type = 'vitals'`.

| Column Name | Description | Unit |
|------------|-------------|------|
| `vitaltime` | Vital sign measurement time | Absolute timestamp |
| `heart_rate` | Heart rate | Beats per minute |
| `sbp` | Systolic blood pressure | mmHg |
| `temperature` | Body temperature | °C |

---

## 7. Vital Sign Warning Labels

Binary warning indicators are included to facilitate downstream risk modeling. These labels are generated based on predefined clinical thresholds.

| Column Name | Description | Values |
|------------|-------------|--------|
| `hr_warn` | Heart rate abnormality | 1 = abnormal, 0 = normal |
| `sbp_warn` | Blood pressure abnormality | 1 = abnormal, 0 = normal |
| `temp_warn` | Temperature abnormality | 1 = abnormal, 0 = normal |

> The exact threshold rules should be specified in the Methods or Experimental Setup section.

---

## 8. Patient Demographics

| Column Name | Description | Notes |
|------------|-------------|-------|
| `gender` | Patient gender | M / F |
| `anchor_age` | Patient age | De-identified age |
| `patientweight` | Body weight | kg |

---

## 9. Admission and ICU Timing

| Column Name | Description | Notes |
|------------|-------------|-------|
| `intime` | Admission / ICU entry time | Reference time point |
| `outtime` | Discharge / ICU exit time | May be missing |

All relative time variables (`rel_time`, `vanco_start_rel`, `vanco_end_rel`) are calculated with respect to `intime`.

---

## 10. Missing Values and Data Sparsity

- Only variables relevant to a given event type are populated; all others are missing
- Lab tests, vital signs, and medication records occur at irregular time intervals
- The dataset represents **sparse, multivariate clinical time-series data**

Preprocessing steps such as temporal alignment, resampling, aggregation, or interpolation may be required depending on the modeling approach.

---

## 11. Usage Notes

- Group data by `stay_id` before analysis
- Use `rel_time` as the unified temporal axis
- Split or fuse lab, vital, and drug data according to the task
- When describing the dataset in a paper, emphasize:
  - Raw (non-aggregated) nature of the data
  - Irregular temporal structure
  - Integration of multiple clinical modalities

---

**Document type**: Raw Data Description / Data Dictionary
