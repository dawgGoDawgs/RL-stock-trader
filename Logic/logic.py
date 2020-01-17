# Import libraries for trading strategy
from scipy import stats
import numpy as np
from math import *

# Function that contains our investment logic. Edit this to change how the RL brain acts
# This is a simple boilerplate trading strategy that should be modified to yield better returns
# ** If you are changing the trading strategy, keep 0 for Sell and 1 for Buy ** 
def state_logic(pointer, data):
    spot_price = data['EQUITY'][pointer]
    rf_rate = data['RF'][pointer] if not np.isnan(data['RF'][pointer]) else 1.0 # default to 1% if nan
    sigma = data['SIGMA'][pointer]
    # BSM for if the option's exercise price appreciates by 5%
    price_increase = calculate_BSM(spot_price,
                                   spot_price * 1.05, # 5% appreciation
                                   rf_rate / 100,
                                   sigma / 100, 1 # Roughly one day timeframe
                                   / 365.0)
    # BSM for if the option's exercise price holds its value
    stable_price = calculate_BSM(spot_price, spot_price, rf_rate / 100,
                                 sigma / 100, 1
                                 / 365.0)
    # Tinker with this 
    returns = log(stable_price / price_increase)
    # Tinker with the return threshold as well
    print "pointer:", pointer
    if returns <= 10:
        return 1  # sell
    if returns > 10:
        return 0  # buy

# Black-Scholes Model function needed for our current state logic
def calculate_BSM(
    Equity,
    Strike_Price,
    RF_Rate,
    MKT_Vol,
    TimeFrame,
    ):
    d1 = (np.log(Equity / Strike_Price) + (RF_Rate + 0.5 * MKT_Vol
          ** 2) * TimeFrame) / (float(MKT_Vol) * np.sqrt(TimeFrame))
    d2 = (np.log(Equity / Strike_Price) + (RF_Rate - 0.5 * MKT_Vol
          ** 2) * TimeFrame) / (float(MKT_Vol) * np.sqrt(TimeFrame))
    value = Equity * stats.norm.cdf(d1, 0.0, 1.0) - Strike_Price \
        * np.exp(-RF_Rate * TimeFrame) * stats.norm.cdf(d2, 0.0, 1.0)
    return value
