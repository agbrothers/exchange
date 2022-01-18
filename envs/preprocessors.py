import numpy as np
import pandas as pd


class StockPreprocessor: 
    # Preprocess listing obs to pass relative values to agents

    def __init__(self, ics, dt=1):
        self.ticker = ""
        self.obs = {}
        self.timestep = -1 
        self.dt = dt    

        # Histories
        self.price_history = []
        self.price_percentage_history = []
        self.volume_history = [] 
        self.volume_percentage_history = []

        # Intraday price high
        self.obs["day_price_high"] = 0
        # Intraday price low
        self.obs["day_price_low"] = np.inf

        # Intraday volume high
        self.obs["day_volume_high"] = 0
        # Intraday low
        self.obs["day_volume_low"] = np.inf

        self.update(ics)


    def update(self, obs): 
        self.timestep += 1

        # current time interval share price open
        self.obs["current_price_open"] = obs[0]
        # current time interval share price high
        self.obs["current_price_high"] = obs[1]
        # current time interval share price low
        self.obs["current_price_low"] = obs[2]
        # current time interval share price close
        self.obs["current_price_close"] = obs[3]
        # current time interval share volume
        self.obs["current_volume"] = obs[4]

        # current price percentage change
        self.obs["current_price_percentage_change"] = self.obs["current_price_close"] / self.obs["current_price_open"] - 1
        # current price percentage high
        self.obs["current_price_percentage_high"]   = self.obs["current_price_close"] / self.obs["current_price_high"] - 1
        # current price percentage low
        self.obs["current_price_percentage_low"]    = self.obs["current_price_close"] / self.obs["current_price_low"] - 1

        self.price_history.append(self.obs["current_price_close"])
        self.price_percentage_history.append(self.obs["current_price_percentage_change"])

        # Intraday price open
        if self.timestep==0: self.obs["day_price_open"] = self.obs["current_price_open"]
        # Intraday price high
        self.obs["day_price_high"]    = max(self.obs["current_price_high"], self.obs["day_price_high"])
        # Intraday price low
        self.obs["day_price_low"]     = min(self.obs["current_price_low"], self.obs["day_price_low"])
        # Intraday price avg
        self.obs["day_price_median"]  = np.median(self.price_history[::self.dt])
        # Intraday price 30m moving avg
        self.obs["day_price_30m_avg"] = np.mean(self.price_history[-30::self.dt])

        # Intraday price percentage change
        self.obs["day_price_percentage_change"]  = self.obs["current_price_close"] / self.obs["day_price_open"] - 1
        # Intraday price percentage high
        self.obs["day_price_percentage_high"]    = self.obs["current_price_close"] / self.obs["day_price_high"] - 1
        # Intraday price percentage low
        self.obs["day_price_percentage_low"]     = self.obs["current_price_close"] / self.obs["day_price_low"] - 1
        # Intraday price avg
        self.obs["day_price_median_percentage"]  = np.median(self.price_percentage_history[::self.dt])
        # Intraday price 30m moving avg
        self.obs["day_price_30m_avg_percentage"] = np.mean(self.price_percentage_history[-30::self.dt])

        self.volume_history.append(self.obs["current_volume"])
        self.volume_percentage_history.append(self.obs["current_volume"] / self.volume_history[self.timestep-1] - 1)

        # Intraday volume open
        if self.timestep==0: self.obs["day_volume_open"] = self.obs["current_volume"]
        # Intraday high
        self.obs["day_volume_high"] = max(self.obs["current_volume"], self.obs["day_volume_high"])
        # Intraday low
        self.obs["day_volume_low"] = min(self.obs["current_volume"], self.obs["day_volume_low"])
        # Intraday open
        self.obs["day_volume_avg"] = np.median(self.volume_history[::self.dt])
        # Intraday open
        self.obs["day_volume_30m_avg"] = np.mean(self.volume_history[-30::self.dt])

        # Intraday price percentage change
        self.obs["day_volume_percentage_change"]  = self.obs["current_volume"] / self.obs["day_volume_open"] - 1
        # Intraday price percentage high
        self.obs["day_volume_percentage_high"]    = self.obs["current_volume"] / self.obs["day_volume_high"] - 1
        # Intraday price percentage low
        self.obs["day_volume_percentage_low"]     = self.obs["current_volume"] / self.obs["day_volume_low"] - 1
        # Intraday price avg
        self.obs["day_volume_median_percentage"]  = np.median(self.volume_percentage_history[::self.dt])
        # Intraday price 30m moving avg
        self.obs["day_volume_30m_avg_percentage"] = np.mean(self.volume_percentage_history[-30::self.dt])


    def get_state(self):
        return self.obs



if __name__ == "__main__":

    columns = ["date", "open", "high", "low", "close", "volume"]
    ticker_paths = {
        "AAPL": "AAPL_1min_sample.csv",
        "MSFT": "MSFT_1min_sample.csv",
    }
    ticker = "AAPL"
    i = 0
    data = pd.read_csv(f"data/{ticker_paths[ticker]}", names=columns)
    data[["date","time"]] = data.date.str.split(" ", expand=True)
    date_list = list(data.groupby("date").groups.keys())
    date = np.random.choice(date_list)
    
    time_series = data.groupby("date").get_group(date)[columns[1:]].values
    ics = time_series[i]
    test = StockPreprocessor(ics)

    import pprint
    pp = pprint.PrettyPrinter(indent=4)

    for i in range(1,len(time_series)):
        obs = time_series[i]
        test.update(obs)
        new = test.get_state()
        # print(new)
        pp.pprint(new)


    
    print("okay")