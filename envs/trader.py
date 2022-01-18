
import coax
from gym.spaces import Box


# Preprocess the observations
# House the RL algorithm
# Provide short-term memory for important intraday statistics

    # NOTE: Could have continuous action space for buying fractional shares
    # How will buying/selling work with discrete actions?
        # Should have continuous action spaces [-1,1] buy / sell that % of the balance or value
        # Will need a action for each available stock
    # self.actions = Box(0, np.inf, shape=(3)) # buy $N, sell $N, hold
    # self.obs = Box(-np.inf, np.inf, shape=(6)) # open, high, low, close, volume, balance, % change



class Trader:
    # Convert Actions into trades

    def __init__(self) -> None:
        self.actions = Box(-1, 1, shape=[len(self.tickers)]) # sell -1 > 0, hold == 0, buy 0 < 1        
        self.portfolio=[]


    def get_state(self, obs):
        

        processed_obs = {}
        processed_obs["current_step"] = self.current_step
        processed_obs["current_cash_balance"] = self.current_cash_balance
        processed_obs["current_portfolio_value"] = self.current_portfolio_value
        processed_obs["current_stock_value"] = self.starting_balance

        # convert the current share price to the fraction of the cash balance
        # prevent it from having cash balance < 1
        # run into the issue of the fraction blowing up when the balance dips below the stock price
            # which defeats the purpose of normalizing it in that way
            # but still may be useful?


        return 


