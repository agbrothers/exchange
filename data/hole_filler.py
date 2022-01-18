

"""
Data README: 
- Times with zero volume are omitted (thus gaps in the data sequence are when there have been no trades)

Fix this ^

Then 
    self.data = pd.read_csv(f"data/{ticker_paths[ticker]}", names=columns)
    self.data[["date","time"]] = self.data.date.str.split(" ", expand=True)


- Volume is not available for indices
- Timezone is US Eastern Time   
- 4:00 - 20:00
- make duplicates with time constrained to robinhood's trading hours

"""
import datetime
import pandas as pd
from glob import glob

if __name__ == "__main__":

    ticker_paths = {
        "AAPL": "AAPL_1min_sample.csv",
        "MSFT": "MSFT_1min_sample.csv",
    }

    # paths = glob("data/*.csv")

    for path in ticker_paths:


        path = "data/MSFT_1min_sample.csv"
        data = pd.read_csv(path)

        # # Separate dates and times
        # data[["date","time"]] = data.date.str.split(" ", expand=True)
        # data.to_csv(path, index=False)


        # Forward fill in missing minutes 
        data['time'] = pd.to_datetime(data['time'])

        i = 1
        num_rows = data.shape[0]
        while i < num_rows:
            print(i, num_rows)
            # Inject if difference > 1 sec
            # recusrive with new injections
            t0 = data.iloc[i-1]["time"]
            t1 = data.iloc[i]["time"]
            difference = t1 - t0
            if difference.total_seconds() > 60 and difference.total_seconds() < 60*60:
                print(difference.total_seconds())
                print(t0,t1)
                minute = (t0.minute + 1) % 60
                t_new = t0.replace(minute=minute)
                new_row = pd.DataFrame(data.iloc[i-1].copy()).T
                new_row["time"] = t_new
                new_row["volume"] = 0
                new_row["open"] = new_row["close"]
                new_row["high"] = new_row["close"]
                new_row["low"] = new_row["close"]

                data = pd.concat([data.iloc[:i], new_row, data.iloc[i:]]).reset_index(drop=True)
                num_rows += 1

            i+=1


        data['time'] = data['time'].astype("string")
        data["time"] = data.time.str.split(" ", expand=True)[1]

        data.to_csv(path, index=False)
