import os, json, joblib, numpy as np, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier
from f1_project.models.metrics import race_topk_metrics, ranking_mrr, ranking_ndcg_at_k, global_binary_metrics, plot_calibration, plot_lift_deciles

PRE_RACE_FEATURES = [
    "GridPosition","qual_position","qual_gap_to_pole_s",
    "sprint_position","sprint_points",
    "form_points_mean_3","form_points_sum_5","form_podium_rate_5","form_dnf_rate_5",
    "form_avg_finish_pos_3","form_avg_qual_pos_3","form_avg_grid_pos_3","form_delta_grid_finish_mean_3",
    "team_points_mean_3","team_points_sum_5","team_podium_rate_5","team_dnf_rate_5",
    "team_avg_finish_pos_3","team_avg_qual_pos_3","team_avg_grid_pos_3",
]

def _default_pipe():
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("clf", HistGradientBoostingClassifier(max_iter=800, learning_rate=0.05,
                                               max_depth=6, l2_regularization=0.1, random_state=42))
    ])

def train_eval(csv_path:str, out_dir:str="reports", save_model:bool=False, artifacts_dir:str="artifacts"):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(csv_path, parse_dates=["race_date_utc"])
    df["DriverNumber"] = df["DriverNumber"].astype(str)
    df["driver_tag"] = (df["Abbreviation"].fillna(df["BroadcastName"]).fillna(df["DriverId"]).fillna("#"+df["DriverNumber"]))
    df["race_id"] = df["season"].astype(str) + "-" + df["round"].astype(str)

    feats = [c for c in PRE_RACE_FEATURES if c in df.columns]
    X, y = df[feats], df["is_winner"].astype(int).values

    # Hold out the most recent race
    t = pd.to_datetime(df["race_date_utc"], utc=True, errors="coerce")
    last_ts = t.max()
    test_mask  = (t == last_ts)
    train_mask = (t <  last_ts)

    n_tr, n_te = int(train_mask.sum()), int(test_mask.sum())
    if n_tr == 0 or n_te == 0:
        raise ValueError(f"Empty split: train={n_tr}, test={n_te}. Check CSV.")

    Xtr, ytr = X[train_mask], y[train_mask]
    Xte, yte = X[test_mask],  y[test_mask]
    meta_te  = df.loc[test_mask, ["race_id","season","round","driver_tag","TeamName"]].copy()

    # simple class-imbalance weighting per race (optional but helps)
    race_key_tr = df.loc[train_mask, "race_id"]
    n_per = race_key_tr.map(race_key_tr.value_counts()).values
    w = np.where(ytr == 1, n_per - 1, 1.0)

    pipe = _default_pipe()
    pipe.fit(Xtr, ytr, clf__sample_weight=w)
    p = pipe.predict_proba(Xte)[:,1]

    pred = meta_te.copy()
    pred["p_win"] = p; pred["y_true"] = yte

    os.makedirs(f"{out_dir}/metrics", exist_ok=True)
    os.makedirs(f"{out_dir}/figures", exist_ok=True)
    pred.sort_values(["season","round","p_win"], ascending=[True,True,False]).to_csv(f"{out_dir}/metrics/test_rankings_latest.csv", index=False)

    kmet = race_topk_metrics(pred, k_list=(1,3,5))
    mrr = ranking_mrr(pred); ndcg3 = ranking_ndcg_at_k(pred, k=3)
    gmet = global_binary_metrics(pred["y_true"].values, pred["p_win"].values)
    plot_calibration(pred["y_true"].values, pred["p_win"].values, f"{out_dir}/figures/calibration.png")
    plot_lift_deciles(pred["y_true"].values, pred["p_win"].values, f"{out_dir}/figures/lift_deciles.png")
    pd.Series({**kmet, "mrr": mrr, "ndcg@3": ndcg3, **gmet, "n_races": pred["race_id"].nunique()}).to_csv(f"{out_dir}/metrics/summary.csv")

    if save_model:
        os.makedirs(artifacts_dir, exist_ok=True)
        joblib.dump(pipe, os.path.join(artifacts_dir, "model.joblib"))
        with open(os.path.join(artifacts_dir, "features.json"), "w") as f:
            json.dump({"features": feats}, f, indent=2)
        print(f"Saved model -> {artifacts_dir}/model.joblib and features.json")

    return pipe, feats