import os, argparse, pandas as pd, numpy as np, time
from collections import defaultdict, Counter
from typing import List, Dict
import fastf1
import sys
sys.path.insert(1, '/Users/brynm/Desktop/codeProject/f1-project-new/src/f1_project/utils/')

from f1_project.utils.features import add_rolling_form_features, add_constructor_form_features

def summarize_weather(weather_df: pd.DataFrame) -> Dict[str, float]:
    if weather_df is None or weather_df.empty:
        return {k: np.nan for k in [
            "AirTemp_mean","AirTemp_min","AirTemp_max",
            "TrackTemp_mean","TrackTemp_min","TrackTemp_max",
            "WindSpeed_mean","Rainfall_rate"
        ]}
    def safe_mean(x): return float(np.nanmean(x)) if x is not None else np.nan
    def safe_min(x):  return float(np.nanmin(x))  if x is not None else np.nan
    def safe_max(x):  return float(np.nanmax(x))  if x is not None else np.nan
    def rain_rate(x):
        if x is None or len(x) == 0: return np.nan
        return float(np.mean((x > 0).astype(float)))
    AirTemp = weather_df.get("AirTemp"); TrackTemp = weather_df.get("TrackTemp")
    WindSpeed = weather_df.get("WindSpeed"); Rainfall = weather_df.get("Rainfall")
    return {
        "AirTemp_mean": safe_mean(AirTemp), "AirTemp_min": safe_min(AirTemp), "AirTemp_max": safe_max(AirTemp),
        "TrackTemp_mean": safe_mean(TrackTemp), "TrackTemp_min": safe_min(TrackTemp), "TrackTemp_max": safe_max(TrackTemp),
        "WindSpeed_mean": safe_mean(WindSpeed), "Rainfall_rate": rain_rate(Rainfall)
    }

def get_qual_features(year:int, rnd:int) -> pd.DataFrame:
    try:
        q = fastf1.get_session(year, rnd, "Q"); q.load(laps=False, telemetry=False, weather=False, messages=False)
        res = q.results
        if res is None or len(res)==0: return pd.DataFrame()
    except Exception: return pd.DataFrame()
    df = res.copy()
    qcols = [c for c in ["Q1","Q2","Q3"] if c in df.columns]
    for c in qcols: df[c] = pd.to_timedelta(df[c], errors="coerce")
    df["BestQualTime"] = df[qcols].min(axis=1, skipna=True) if qcols else pd.NaT
    df["qual_position"] = df["Position"]
    if not df["BestQualTime"].isna().all():
        pole = df.loc[df["qual_position"] == 1, "BestQualTime"].iloc[0]
        df["qual_gap_to_pole_s"] = (df["BestQualTime"] - pole).dt.total_seconds()
    else:
        df["qual_gap_to_pole_s"] = np.nan
    keep = ["DriverNumber","qual_position","qual_gap_to_pole_s"]
    df = df[[c for c in keep if c in df.columns]]
    df["season"], df["round"] = year, rnd
    df["DriverNumber"] = df["DriverNumber"].astype(str)
    return df

def get_event_ts(year:int, event_query:str):
    sched = fastf1.get_event_schedule(year)
    mask = (
        sched["EventName"].str.contains(event_query, case=False, na=False) |
        sched["Country"].str.contains(event_query, case=False, na=False) |
        sched["Location"].str.contains(event_query, case=False, na=False)
    )
    row = sched[mask]
    if row.empty:
        raise ValueError(f'Could not find event matching "{event_query}" in {year}.')
    r = row.iloc[0]
    raw = r.get("Session5DateUtc") or r.get("Session5Date") or r.get("EventDate")
    ts  = pd.to_datetime(raw, utc=True, errors="coerce")
    rnd = int(r["RoundNumber"])
    return ts, rnd




def get_sprint_features(year:int, rnd:int) -> pd.DataFrame:
    try:
        s = fastf1.get_session(year, rnd, "S"); s.load(laps=False, telemetry=False, weather=False, messages=False)
        res = s.results
        if res is None or len(res)==0: return pd.DataFrame()
    except Exception: return pd.DataFrame()
    df = res.copy().rename(columns={"Position":"sprint_position","Points":"sprint_points","Status":"sprint_status"})
    keep = ["DriverNumber","sprint_position","sprint_points","sprint_status"]
    df = df[[c for c in keep if c in df.columns]]
    df["season"], df["round"] = year, rnd
    df["DriverNumber"] = df["DriverNumber"].astype(str)
    return df

def get_hungary_2025_cutoff_utc() -> pd.Timestamp | None:
    try:
        sched = fastf1.get_event_schedule(2025)
        mask = (sched.get("Country", pd.Series(dtype=str)).fillna("").str.lower()=="hungary") | \
               (sched.get("EventName", pd.Series(dtype=str)).fillna("").str.contains("Hungarian", case=False))
        row = sched[mask]
        if row.empty: return None
        r = row.iloc[0]
        raw = r.get("Session5DateUtc") or r.get("Session5Date") or r.get("EventDate")
        return pd.to_datetime(raw, utc=True, errors="coerce")
    except Exception:
        return None

def load_session_with_retry(year:int, rnd:int, dtype:str="R", want_weather:bool=True, retries:int=3, backoff:float=0.75):
    last = None
    for i in range(retries):
        try:
            sess = fastf1.get_session(year, rnd, dtype)
            sess.load(laps=False, telemetry=False, weather=want_weather, messages=False)
            return sess
        except Exception as e:
            last = e; time.sleep(backoff*(i+1)); want_weather=False
    print(f"[{year} R{rnd} {dtype}] load failed after retries: {last}")
    return None

def build_dataset(years:List[int]) -> pd.DataFrame:
    rows, skips, skipped_events = [], defaultdict(int), []
    cutoff = get_hungary_2025_cutoff_utc()
    for year in years:
        try:
            schedule = fastf1.get_event_schedule(year)
        except Exception as e:
            print(f"[{year}] schedule error: {e}"); continue
        for _, evt in schedule.iterrows():
            try:
                rnd = int(evt["RoundNumber"]); name = evt["EventName"]; country = evt["Country"]
                if rnd==0 or any(k in str(name).lower() for k in ["test","testing","track session","pre-season"]):
                    skips["non_race"]+=1; skipped_events.append((year,rnd,name,"non_race")); continue
                # Parse race date
                race_date = evt.get("Session5DateUtc") or evt.get("Session5Date") or evt.get("EventDate")
                race_ts = pd.to_datetime(race_date, utc=True, errors="coerce") if race_date is not None else pd.NaT
                sess = load_session_with_retry(year, rnd, "R", want_weather=True)
                if sess is None: skips["load_failed"]+=1; skipped_events.append((year,rnd,name,"load_failed")); continue
                res = sess.results
                if res is None or len(res)==0: skips["no_results"]+=1; skipped_events.append((year,rnd,name,"no_results")); continue
                # Winner
                winner_abbr = None
                for pos_col in ("Position","ClassifiedPosition","FinalPosition"):
                    if pos_col in res.columns:
                        pos = pd.to_numeric(res[pos_col], errors="coerce")
                        if pos.notna().any() and (pos==1).any():
                            winner_abbr = res.loc[pos==1,"Abbreviation"].dropna().astype(str).iloc[0]; break
                if winner_abbr is None and "Points" in res.columns:
                    pts = pd.to_numeric(res["Points"], errors="coerce")
                    if pts.notna().any():
                        cand = res.loc[pts==pts.max(skipna=True)].copy()
                        for pos_col in ("Position","ClassifiedPosition","FinalPosition"):
                            if pos_col in cand.columns:
                                posn = pd.to_numeric(cand[pos_col], errors="coerce")
                                if posn.notna().any(): winner_abbr = cand.loc[posn.idxmin(),"Abbreviation"]; break
                        if winner_abbr is None: winner_abbr = cand["Abbreviation"].dropna().astype(str).iloc[0]
                w = summarize_weather(sess.weather_data)
                tmp = res.copy()
                tmp["season"], tmp["round"] = year, rnd
                tmp["event_name"], tmp["country"] = name, country
                tmp["race_date_utc"] = race_ts
                if "Laps" in res.columns:
                    laps_series = pd.to_numeric(res["Laps"], errors="coerce")
                    tmp["race_laps_completed"] = float(laps_series.max()) if laps_series.notna().any() else np.nan
                else: tmp["race_laps_completed"] = np.nan
                tmp["is_winner"] = (tmp["Abbreviation"] == winner_abbr).astype(int) if winner_abbr else 0
                for k,v in w.items(): tmp[k]=v
                tmp["DriverNumber"] = tmp["DriverNumber"].astype(str)
                q = get_qual_features(year, rnd)
                if (not q.empty) and all(c in q.columns for c in ["season","round","DriverNumber"]):
                    tmp = tmp.merge(q, on=["season","round","DriverNumber"], how="left")
                
                s = get_sprint_features(year, rnd)
                if (not s.empty) and all(c in s.columns for c in ["season","round","DriverNumber"]):
                    tmp = tmp.merge(s, on=["season","round","DriverNumber"], how="left")
            
                keep = ["season","round","event_name","country","race_date_utc",
                        "DriverNumber","DriverId","BroadcastName","Abbreviation","TeamName",
                        "GridPosition","Position","Status","Points","race_laps_completed","is_winner",
                        "AirTemp_mean","AirTemp_min","AirTemp_max","TrackTemp_mean","TrackTemp_min","TrackTemp_max",
                        "WindSpeed_mean","Rainfall_rate","qual_position","qual_gap_to_pole_s",
                        "sprint_position","sprint_points","sprint_status"]
                for c in keep:
                    if c not in tmp.columns: tmp[c]=np.nan
                rows.append(tmp[keep])
            except Exception as e:
                skips["exception"]+=1; skipped_events.append((year,rnd,name,f"exception:{e}")); continue
    if not rows: return pd.DataFrame()
    df = pd.concat(rows, ignore_index=True)
    if cutoff is not None:
        before = df.shape[0]
        mask = ~((df["season"]==2025) & (pd.to_datetime(df["race_date_utc"], utc=True, errors="coerce") >= cutoff))
        df = df[mask].copy()
        print(f"Filtered 2025>=Hungary rows: {before - df.shape[0]}")
    
    if os.environ.get("UPTO_YEAR") and os.environ.get("UPTO_EVENT"):
        upto_year  = int(os.environ["UPTO_YEAR"])
        upto_event = os.environ["UPTO_EVENT"]
        ts, rnd = get_event_ts(upto_year, upto_event)
        before = df.shape[0]
        df = df[pd.to_datetime(df["race_date_utc"], utc=True, errors="coerce") <= ts].copy()
        print(f'Cut to ≤ {upto_event} {upto_year} (Round {rnd}). Dropped {before - df.shape[0]} rows.')
    
    
    
    
    df = df.sort_values(["season","round"]).reset_index(drop=True)
    print("Per-season rows:", df["season"].value_counts().sort_index().to_dict())
    # numerical coercions
    for c in ["GridPosition","Position","Points","race_laps_completed",
              "qual_position","qual_gap_to_pole_s","sprint_position","sprint_points"]:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
    df = add_rolling_form_features(df)
    df = add_constructor_form_features(df)
    

    #DEBUG
    '''
    if skipped_events:
        reasons_all = Counter([reason for (_, _, _, reason) in skipped_events])
        print("Skipped events by reason:", dict(reasons_all))

    sk_2025 = [(yr, rnd, name, reason) for (yr, rnd, name, reason) in skipped_events if yr == 2025]
    if sk_2025:
        reasons_2025 = Counter([reason for (_, _, _, reason) in sk_2025])
        print("Skipped 2025 events by reason:", dict(reasons_2025))
        print("Details (2025):")
        for yr, rnd, name, reason in sk_2025:
            print(f"  {yr} R{rnd} {name}: {reason}")
    '''
    return df

def cli():
    p = argparse.ArgumentParser()
    p.add_argument("--years", nargs="+", type=int, default=list(range(2020, 2026)))
    p.add_argument("--out", default="data/f1_driver_race_with_qual_sprint_teams_2020_2025preHUN.csv")
    p.add_argument("--cache", default="cache_folder")
    args = p.parse_args()
    os.makedirs(args.cache, exist_ok=True)
    fastf1.Cache.enable_cache(args.cache)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df = build_dataset(args.years)
    df.to_csv(args.out, index=False)
    print(f"Saved dataset -> {args.out} with shape {df.shape}")

if __name__ == "__main__":
    cli()