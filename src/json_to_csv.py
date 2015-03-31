import pandas as pd
import json
import os
from glob import glob

if __name__ == "__main__":
    for results_path in glob("results/*AIA*_results.json"):
        with open(results_path, "r") as f:
            results = json.load(f)
        for measure in ("tss", "tpr", "fpr"):
            df = pd.DataFrame(results[measure])
            df.to_csv("{}_{}.csv".format(os.path.splitext(results_path)[0], measure))
