import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.metrics import log_loss, roc_auc_score, average_precision_score, brier_score_loss

def race_topk_metrics(df_pred: pd.DataFrame, k_list=(1,3,5)):
    """df_pred has columns: race_id, y_true, p_win"""
    out = {}
    for k in k_list:
        hits = 0; races = 0
        for rid, g in df_pred.groupby("race_id"):
            g = g.sort_values("p_win", ascending=False)
            if (g.head(k)["y_true"]==1).any(): hits += 1
            races += 1
        out[f"top{k}_acc"] = hits / max(races,1)
    return out

def ranking_mrr(df_pred: pd.DataFrame):
    mrr = []
    for rid, g in df_pred.groupby("race_id"):
        g = g.sort_values("p_win", ascending=False).reset_index(drop=True)
        pos = g.index[g["y_true"]==1]
        if len(pos): mrr.append(1.0/(pos[0]+1))
    return float(np.mean(mrr)) if mrr else np.nan

def ranking_ndcg_at_k(df_pred: pd.DataFrame, k=3):
    import math
    ndcgs = []
    for rid, g in df_pred.groupby("race_id"):
        g = g.sort_values("p_win", ascending=False)
        rel = g["y_true"].values[:k]
        dcg = sum((rel[i]/math.log2(i+2) for i in range(len(rel))))
        idcg = 1.0  # only one winner
        ndcgs.append(dcg/idcg)
    return float(np.mean(ndcgs)) if ndcgs else np.nan

def global_binary_metrics(y_true, p):
    out = {}
    mask = ~np.isnan(p)
    y = np.array(y_true)[mask]; s = np.array(p)[mask]
    if len(np.unique(y)) == 1:
        out.update({"logloss": np.nan, "roc_auc": np.nan, "pr_auc": np.nan, "brier": np.nan})
    else:
        out["logloss"] = log_loss(y, s, labels=[0,1])
        out["roc_auc"] = roc_auc_score(y, s)
        out["pr_auc"]  = average_precision_score(y, s)
        out["brier"]   = brier_score_loss(y, s)
    return out

def plot_calibration(y_true, p, outpath):
    bins = np.linspace(0,1,11)
    df = pd.DataFrame({"y":y_true, "p":p}).dropna()
    df["bin"] = pd.cut(df["p"], bins, include_lowest=True)
    cal = df.groupby("bin").agg(mean_p=("p","mean"), win_rate=("y","mean")).reset_index(drop=True)
    plt.figure()
    plt.plot([0,1],[0,1], linestyle="--")
    plt.scatter(cal["mean_p"], cal["win_rate"])
    plt.xlabel("Predicted win prob"); plt.ylabel("Observed win rate"); plt.title("Calibration")
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.savefig(outpath, bbox_inches="tight"); plt.close()

def plot_lift_deciles(y_true, p, outpath):
    df = pd.DataFrame({"y":y_true, "p":p}).dropna().sort_values("p", ascending=False).reset_index(drop=True)
    df["decile"] = pd.qcut(df["p"], 10, labels=False, duplicates="drop")
    lift = df.groupby("decile").agg(avg_p=("p","mean"), win_rate=("y","mean")).reset_index()
    plt.figure()
    plt.plot(lift["decile"], lift["win_rate"], marker="o")
    plt.xlabel("Decile (0=highest)"); plt.ylabel("Observed win rate"); plt.title("Lift Chart")
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.savefig(outpath, bbox_inches="tight"); plt.close()