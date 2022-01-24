

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
import os
import time
import datetime
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm




def trim_hours(df):
    now = datetime.datetime.now().time()
    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0) # 9:30 am EST
    market_close = now.replace(hour=16, minute=0, second=0, microsecond=0) # 4:00 pm EST

    data = df.copy()
    print(data.shape)
    data = data[data["time"] >= market_open]
    data = data[data["time"] < market_close]
    print(data.shape)
    
    return data


def process_csv(path):
    columns = ["date","open","high","low","close","volume"]
    data = pd.read_csv(path,columns)

    data['timestamp'] = pd.to_datetime(data['date'])
    data['time'] = data.timestamp.dt.time
    data = trim_hours(data)

    data['date'] = data.timestamp.dt.date
    data = data.set_index('timestamp')
    filled = data.groupby('date').apply(lambda x: x.resample('1Min').asfreq()).reset_index('date', drop=True) 
    volume = filled["volume"].fillna(0)
    data = filled.ffill()
    data["volume"] = volume
    data = data.reset_index('timestamp') 
    data['time'] = data.timestamp.dt.time
    rows_filled = data[data["volume"]==0].shape
    print(path, rows_filled)

    data.to_csv(path, index=False)


if __name__ == "__main__":
    paths = glob("data/source/*.csv")
    for path in tqdm(paths):
        process_csv(path)



    # data[data["date"]=="2005-11-25"].head(50)
    # data[data["date"]=="2005-11-25"].tail(50)
    # eval = data.groupby('date').count()
    # eval[eval["open"]!=390]
    # # 211 -> 1pm
    # data[data["date"]=="2021-11-26"]


    # def to_sec(time):
    #     seconds = (time-datetime.datetime(1970,1,1)).total_seconds()
    #     return seconds

    # data['time'] = data.timestamp.dt.time.astype(str)
    # nows = pd.to_datetime(data['time'])
    # test = nows.apply(to_sec)
    # t0 = test[0]
    # to_index = lambda tn: int((tn % t0) / 60)
    # indices = test.apply(to_index)
    # data["modulo"] = indices

    # diff = indices.iloc[1:].values - indices.iloc[:-1].values
    # diff = np.concatenate(([1],diff))
    # diff[diff>1]
    # data["diff"] = diff

    # now = datetime.time.now()

    # ls = data.timestamp.dt.time
    # ls[0] > ls[1]


    # now = datetime.datetime.now()
    # now_time = now.time()
    # ls = data.timestamp.dt.time

    # market_open = now_time.replace(hour=9, minute=30, second=0, microsecond=0) # 9:30 am EST
    # market_close = now_time.replace(hour=16, minute=0, second=0, microsecond=0) # 4:00 pm EST

    # ls > market_open
