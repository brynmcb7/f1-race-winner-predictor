# F1 Race Winner Predictor

Small end-to-end pipeline to:
1) **Build** a dataset from FastF1,
2) **Train & evaluate** a simple model,
3) **Predict** a race and output a clean table (predicted order, actual finish, quali).

Data source: [FastF1](https://theoehrly.github.io/Fast-F1/). Very recent races might not have official `results` yet; those will show `actual_finish = NA`.

---

## Quickstart

### 0) Create & activate a virtual env
```bash
python -m venv .venv
# Windows
.\.venv\Scripts\Activate.ps1
# macOS/Linux
source .venv/bin/activate
```


### 1) Install 
```bash    
python -m pip install -U pip
python -m pip install -e .
```
### 2) Build Data
```bash
python main.py build-data --years 2020 2021 2022 2023 2024 2025 --cache cache --out data/f1_driver_race_full_2020_2025.csv
```

### 3) Train and Evaluate, maybe save 
```bash
python main.py train-eval --csv data/f1_driver_race_full_2020_2025.csv --save-model --artifacts artifacts
```

### 4) Predict a Race 
```bash
python main.py predict \
  --csv data/f1_driver_race_full_2020_2025.csv \
  --year 2025 --event "Hungarian Grand Prix" \
  --model artifacts/model.joblib \
  --out_csv reports/metrics/hun2025_prediction.csv \
  --out_html reports/metrics/hun2025_prediction.html
```


---

## Project Structure 


├─ main.py # entrypoint: python main.py <command> ...
├─ pyproject.toml      #project/deps
├─ README.md
├─ LICENSE
├─ src/
│ └─ f1_project/
│ ├─ init.py
│ ├─ data/
│ │ └─ build_dataset.py
│ ├─ models/
│ │ ├─ train_eval.py
│ │ ├─ predict.py
│ │ └─ metrics.py
│ └─ utils/
│ └─ features.py
├─ artifacts/          #saved model + metadata (model.joblib, features.json)
├─ data/               #built CSVs (generated, gitignored)
├─ reports/            #metrics/figures/prediction tables (generated, gitignored)
│ ├─ figures/
│ └─ metrics/
└─ cache/              #FastF1 cache (generated, gitignored)





