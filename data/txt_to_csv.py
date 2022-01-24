import os
import pandas as pd
from glob import glob


if __name__ == "__main__":
    
    files = glob("data/nasdaq100_1min_lcn14h/*.txt")

    for file in files:
        df = pd.read_csv(file)
        new_path = "data/csv/" + os.path.basename(file).replace(".txt",".csv")
        df.to_csv(new_path, index=False)

