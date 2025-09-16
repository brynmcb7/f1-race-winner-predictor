import argparse, os, pandas as pd, fastf1
from f1_project.data.build_dataset import build_dataset
from f1_project.models.train_eval import train_eval
from f1_project.models.predict import predict_race_table

def main():
    ap = argparse.ArgumentParser(prog="f1-project-new")
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build-data")
    b.add_argument("--years", nargs="+", type=int, required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--cache", default="cache")

    e = sub.add_parser("train-eval")
    e.add_argument("--csv", required=True)
    e.add_argument("--out_dir", default="reports")
    e.add_argument("--save-model", action="store_true")
    e.add_argument("--artifacts", default="artifacts")

    p = sub.add_parser("predict")
    p.add_argument("--csv", required=True)
    p.add_argument("--year", type=int, required=True)
    p.add_argument("--event", type=str, required=True)  # e.g. "Italian Grand Prix"
    p.add_argument("--model", default="artifacts/model.joblib")
    p.add_argument("--out_csv", default="reports/metrics/prediction.csv")
    p.add_argument("--out_html", default="reports/metrics/prediction.html")

    args = ap.parse_args()

    if args.cmd == "build-data":
        os.makedirs(args.cache, exist_ok=True)
        fastf1.Cache.enable_cache(args.cache)
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        df = build_dataset(args.years)
        df.to_csv(args.out, index=False)
        print(f"Saved -> {args.out}, shape={df.shape}")

    elif args.cmd == "train-eval":
        pipe, feats = train_eval(args.csv, out_dir=args.out_dir,
                                 save_model=args.save_model, artifacts_dir=args.artifacts)
        print("Features used:", feats)

    elif args.cmd == "predict":
        df = predict_race_table(csv_path=args.csv, year=args.year, event_query=args.event,
                                model_path=args.model, out_csv=args.out_csv, out_html=args.out_html)
        print(df.head(10).to_string(index=False))