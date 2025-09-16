import pandas as pd
import numpy as np

def add_rolling_form_features(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df.copy()
    out = df.copy()
    out["pos_num"]  = pd.to_numeric(out.get("Position"), errors="coerce")
    out["grid_num"] = pd.to_numeric(out.get("GridPosition"), errors="coerce")
    out["qual_num"] = pd.to_numeric(out.get("qual_position"), errors="coerce")
    out["points"]   = pd.to_numeric(out.get("Points"), errors="coerce")
    out["podium_flag"] = (out["pos_num"] <= 3).astype(float)
    out["dnf_flag"]    = out["pos_num"].isna().astype(float)
    tmp = out.sort_values(["DriverId","season","round"]).copy()
    g = tmp.groupby("DriverId", sort=False)
    tmp["form_points_mean_3"]    = g["points"].transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    tmp["form_points_sum_5"]     = g["points"].transform(lambda s: s.shift(1).rolling(5, min_periods=1).sum())
    tmp["form_podium_rate_5"]    = g["podium_flag"].transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    tmp["form_dnf_rate_5"]       = g["dnf_flag"].transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    tmp["form_avg_finish_pos_3"] = g["pos_num"].transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    tmp["form_avg_qual_pos_3"]   = g["qual_num"].transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    tmp["form_avg_grid_pos_3"]   = g["grid_num"].transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    tmp["delta_sf"] = tmp["pos_num"] - tmp["grid_num"]
    tmp["form_delta_grid_finish_mean_3"] = g["delta_sf"].transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    for c in ["form_points_mean_3","form_points_sum_5","form_podium_rate_5","form_dnf_rate_5",
              "form_avg_finish_pos_3","form_avg_qual_pos_3","form_avg_grid_pos_3",
              "form_delta_grid_finish_mean_3"]:
        out[c] = tmp[c]
    return out.drop(columns=["pos_num","grid_num","qual_num","points","podium_flag","dnf_flag"], errors="ignore")

def add_constructor_form_features(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df.copy()
    out = df.copy()
    work = out.copy()
    pos = pd.to_numeric(work.get("Position"), errors="coerce")
    grid = pd.to_numeric(work.get("GridPosition"), errors="coerce")
    qpos = pd.to_numeric(work.get("qual_position"), errors="coerce")
    pts = pd.to_numeric(work.get("Points"), errors="coerce")
    work["pos_num"], work["grid_num"], work["qual_num"], work["points"] = pos, grid, qpos, pts
    work["podium_flag"] = (pos <= 3).astype(float)
    work["dnf_flag"] = pos.isna().astype(float)
    team_race = (work.groupby(["TeamName","season","round"], as_index=False)
                     .agg(team_points=("points","sum"),
                          team_podium_rate=("podium_flag","mean"),
                          team_dnf_rate=("dnf_flag","mean"),
                          team_avg_finish_pos=("pos_num","mean"),
                          team_avg_qual_pos=("qual_num","mean"),
                          team_avg_grid_pos=("grid_num","mean"))
                ).sort_values(["TeamName","season","round"]).reset_index(drop=True)
    g = team_race.groupby("TeamName", sort=False)
    team_race["team_points_mean_3"]    = g["team_points"].transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    team_race["team_points_sum_5"]     = g["team_points"].transform(lambda s: s.shift(1).rolling(5, min_periods=1).sum())
    team_race["team_podium_rate_5"]    = g["team_podium_rate"].transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    team_race["team_dnf_rate_5"]       = g["team_dnf_rate"].transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    team_race["team_avg_finish_pos_3"] = g["team_avg_finish_pos"].transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    team_race["team_avg_qual_pos_3"]   = g["team_avg_qual_pos"].transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    team_race["team_avg_grid_pos_3"]   = g["team_avg_grid_pos"].transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    rolled = team_race[["TeamName","season","round",
                        "team_points_mean_3","team_points_sum_5","team_podium_rate_5","team_dnf_rate_5",
                        "team_avg_finish_pos_3","team_avg_qual_pos_3","team_avg_grid_pos_3"]]
    return out.merge(rolled, on=["TeamName","season","round"], how="left")