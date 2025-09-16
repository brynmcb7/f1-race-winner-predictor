import os, json, joblib, numpy as np, pandas as pd

def _load_model(model_path:str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}. Run train-eval --save-model first.")
    pipe = joblib.load(model_path)
    feats = None
    feat_json = os.path.join(os.path.dirname(model_path), "features.json")
    if os.path.exists(feat_json):
        with open(feat_json) as f: feats = json.load(f).get("features")
    return pipe, feats

def predict_race_table(csv_path:str, year:int, event_query:str, model_path:str,
                       out_csv:str, out_html:str):
    df = pd.read_csv(csv_path, parse_dates=["race_date_utc"])
    df["DriverNumber"] = df["DriverNumber"].astype(str)
    df["driver_tag"] = (df["Abbreviation"].fillna(df["BroadcastName"]).fillna(df["DriverId"]).fillna("#"+df["DriverNumber"]))
    df["round"] = pd.to_numeric(df["round"], errors="coerce")

    # pick target rows directly from CSV
    key = event_query.lower()
    target = df[(df["season"]==year) & (df["event_name"].str.lower().str.contains(key, na=False))]
    if target.empty:
        raise ValueError(f"Race '{event_query}' {year} not found in {csv_path}.")
    ts = pd.to_datetime(target["race_date_utc"], utc=True, errors="coerce").max()

    test_mask  = (df["season"]==year) & (df["race_date_utc"]==ts) & (df["event_name"]==target["event_name"].iloc[0])
    train_mask = (pd.to_datetime(df["race_date_utc"], utc=True, errors="coerce") < ts)

    # load model + features
    pipe, feats = _load_model(model_path)
    if not feats:
        # conservative fallback: intersect features with columns
        from f1_project.models.train_eval import PRE_RACE_FEATURES
        feats = [c for c in PRE_RACE_FEATURES if c in df.columns]

    Xtr = df.loc[train_mask, feats]
    ytr = df.loc[train_mask, "is_winner"].astype(int).values
    if Xtr.shape[0] == 0:
        raise ValueError("Training slice is empty. Ensure your CSV includes races before the target race.")

    # light weighting (same as train-eval)
    race_key_tr = df.loc[train_mask, "season"].astype(str) + "-" + df.loc[train_mask, "round"].astype(str)
    n_per = race_key_tr.map(race_key_tr.value_counts()).values
    w = np.where(ytr == 1, n_per - 1, 1.0)

    pipe.fit(Xtr, ytr, clf__sample_weight=w)

    Xte = df.loc[test_mask, feats]
    meta = df.loc[test_mask, ["driver_tag","TeamName","season","round","event_name",
                              "qual_position","Position"]].copy()
    if Xte.shape[0] == 0:
        raise ValueError("Test slice is empty. Are you sure this race exists in your CSV?")

    p = pipe.predict_proba(Xte)[:,1]
    # odds-normalize within race for nicer per-race probabilities
    eps = 1e-6
    q = np.clip(p, eps, 1-eps)
    odds = q / (1-q)
    meta["p_win"] = odds / odds.sum()

    meta["pred_rank"] = meta["p_win"].rank(ascending=False, method="first").astype(int)
    meta["actual_finish"] = pd.to_numeric(meta["Position"], errors="coerce").astype("Int64")  # may be <NA> if no results yet
    meta["qual_pos"] = pd.to_numeric(meta["qual_position"], errors="coerce").astype("Int64")
    meta = meta.drop(columns=["Position","qual_position"]).sort_values("pred_rank")

    # clean columns + save
    out = meta[["pred_rank","driver_tag","TeamName","p_win","actual_finish","qual_pos","season","round","event_name"]]
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    out.to_csv(out_csv, index=False)

    # pretty HTML for GitHub viewing
    style = out.copy()
    style["p_win"] = style["p_win"].map(lambda x: f"{x:.3f}")
    html = style.to_html(index=False)
    with open(out_html, "w", encoding="utf-8") as f:
        f.write("<style>table{border-collapse:collapse;font-family:Inter,Arial,sans-serif}"
                "th,td{border:1px solid #ddd;padding:6px 10px;}th{background:#f5f5f5;text-align:left}"
                "tr:nth-child(even){background:#fafafa}</style>\n")
        f.write(html)
    print(f"Saved -> {out_csv} and {out_html}")
    return out