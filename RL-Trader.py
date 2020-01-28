#!/usr/bin/python
# coding=utf-8

# Reinforcement Learning Equity Trader

# Edit these values to change how the RL brain learns
EPSILON = .9
ALPHA = .1
GAMMA = .9
stop_loss = -0.01
slippage = -0.001

# Create agent class
class Agent:
    def __init__(self, alpha_input, epsilon_input, gamma_input):
        self.alpha = alpha_input
        self.epsilon = 1 - epsilon_input
        self.gamma = gamma_input

# Create class object
agent = Agent(EPSILON, ALPHA, GAMMA)

# Import Libraries
import numpy as np
from scipy import stats
import pandas_datareader.data as web
from math import log
import pandas as pd
import sys, time, datetime
from Logic.logic import calculate_BSM, state_logic
from talib import ADX, HT_DCPERIOD, RSI, BETA, CORREL, MFI
from hmmlearn.hmm import GaussianHMM
import matplotlib.pyplot as plt

# Welcome message
print "\nThanks for using the Reinforcement Learning Stock Trader by Matija Krolo. If you experience an error, it is most likely because the Equity/Stock you chose to analyize does not have available data before the date you entered. If you encounter an error, please check Yahoo.com/finance to ensure it is not the case. \n"
time.sleep(1)

# Get passed-in arguments
GIVEN_EQUITY, START_DATE, STARTING_PORTFOLIO_VALUE, TRADES_TO_RUN = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]

# Error check arguments
if len(sys.argv) != 5:
    print "To run: RL-Trader.py [EQUITY] [START DATE - DAY/MONTH/YEAR] [STARTING PORTFOLIO VALUE] [HOW MANY TRADES TO RUN BEFORE REINFORCEMENT LEARNING BEGINS]\nEx. RL-Trader.py F 1/1/2000 1000 200"
    exit()

# Get Equity Data
CURRENT_MONTH = datetime.datetime
# Todo: create datetime function for user inputs on end dates
EQUITY_TRAIN = web.get_data_yahoo(GIVEN_EQUITY, end=START_DATE, start="1/1/2000",
                            interval='d')
EQUITY = web.get_data_yahoo(GIVEN_EQUITY, end='1/1/2020', start=START_DATE,
                            interval='d')
MKT_VOLATIILTY = web.get_data_yahoo('^VIX', end='1/1/2020',
                                    start=START_DATE, interval='d')
RF_Rate = web.get_data_yahoo('^TNX', end='1/1/2020', start=START_DATE,
                             interval='d')

## Calculate HMM States
open_v = EQUITY_TRAIN["Open"].values
close_v = EQUITY_TRAIN["Close"].values
high_v = EQUITY_TRAIN["High"].values
low_v = EQUITY_TRAIN["Low"].values
volume = EQUITY_TRAIN["Volume"].values.astype(float)
pct = pd.Series(close_v).pct_change().values
pct_volume = pd.Series(volume).pct_change().values
# feature engineering
adx = ADX(high_v, low_v, close_v, timeperiod=14)
rsi = RSI(close_v, timeperiod=14)
beta = BETA(high_v, low_v, timeperiod=30)
correl = CORREL(high_v, low_v, timeperiod=30)
mfi = MFI(high_v, low_v, close_v, volume, timeperiod=14)
# Pack diff and volume for training.
X_train = np.column_stack([pct, adx, rsi, beta, correl, mfi])
col_mean = np.nanmean(X_train, axis=0)
inds = np.where(np.isnan(X_train))
X_train[inds] = np.take(col_mean, inds[1])

## Calculate HMM States
open_v = EQUITY["Open"].values
close_v = EQUITY["Close"].values
high_v = EQUITY["High"].values
low_v = EQUITY["Low"].values
volume = EQUITY["Volume"].values.astype(float)
pct = pd.Series(close_v).pct_change().values
pct_volume = pd.Series(volume).pct_change().values
# feature engineering
adx = ADX(high_v, low_v, close_v, timeperiod=14)
rsi = RSI(close_v, timeperiod=14)
beta = BETA(high_v, low_v, timeperiod=30)
correl = CORREL(high_v, low_v, timeperiod=30)
mfi = MFI(high_v, low_v, close_v, volume, timeperiod=14)

# Pack diff and volume for training.
X = np.column_stack([pct, adx, rsi, beta, correl, mfi])
col_mean = np.nanmean(X, axis=0)
inds = np.where(np.isnan(X))
X[inds] = np.take(col_mean, inds[1])
# Make an HMM instance and execute fit
num_components = 4
model = GaussianHMM(n_components=num_components, covariance_type="diag", n_iter=1000).fit(X_train)
hidden_states = model.predict(X)

# Why not edit this?
STATES = 16
# Actions of Q-Table
ACTIONS = ['buy', 'sell']
# Holds total trades that can be made
TOTAL_TRADES = len(EQUITY['Close']) 

# Error Check
if int(TRADES_TO_RUN) > TOTAL_TRADES:
    print "\nThere are only " + str(TOTAL_TRADES) + " trading days available from data, which is greater than the input of " + str(TRADES_TO_RUN) + ". Please try again."
    exit()

# Q-Table generator function
def build_q_table(n_states, actions):
    table = pd.DataFrame(np.zeros((n_states, len(actions))),
                         columns=actions)
    return table

# Create dictionary
compile_data = {
    'EQUITY': EQUITY['Adj Close'],
    'RF': RF_Rate['Adj Close'],
    'SIGMA': MKT_VOLATIILTY['Adj Close'],
    'HIDDEN': hidden_states
    }

# Compile dataframe from dictionary
data = pd.DataFrame(compile_data)

# Agent brain for RL
def choose_trade(pointer, q_table, inPortfolio):
    # Logic is only running
    if pointer < int(TRADES_TO_RUN):
        print ("Reinforcement Learning not initiated yet, Q-Table still building.")
    # Find the trade decision from our trade logic
    analytic_decision = state_logic(pointer, data)
    # Select state from Q-Table
    state_actions = q_table.iloc[select_state(pointer, inPortfolio), :]
    # If the greedy factor is less than a randomly distributed number, if there are no values
    # on the Q-table, or if less than half the possible trades have been run without our trading logic,
    # return our analytical trade logic decision
    if np.random.uniform() > float(agent.epsilon) or np.count_nonzero(state_actions.values) == 0 or pointer < int(TRADES_TO_RUN):
        print "analytical"
        return analytic_decision
    # Otherwise, return what has been working
    else:
        maximum = state_actions.idxmax()
        # set stop loss
        price_cur = data["EQUITY"][pointer]
        ret_cur = (price_cur - priceAtPurchase) / priceAtPurchase
        trigger_stop = ret_cur < stop_loss and inPortfolio
        if str(maximum) == 'sell': or trigger_stop:
            if trigger_stop:
                print "trigger stop"
            return 1
        if str(maximum) == 'buy':
            return 0

# Selects the state on the Q-Table
def select_state(pointer, current_in_portfolio):
    # Get the current hidden state
    current_hidden = data["HIDDEN"][pointer]
    current_price = data["EQUITY"][pointer]

    state = 0
    if current_in_portfolio:

        position_ret = (current_price - priceAtPurchase) / float(priceAtPurchase)
        local_win_rate = sum(np.array(position_ret_curve) > 0) / float(len(position_ret_curve))
        local_sharpe = None
        if len(position_ret_curve) > 1:
            local_sharpe = np.array(position_ret_curve).mean() / np.array(position_ret_curve).std()
        print "local_sharpe:", local_sharpe

        if current_hidden == 0:
            state = 0 # Equity Appreciated and Hidden is 0
        if current_hidden == 1:
            state = 1 # Equity Appreciated and Hidden is 1
        if current_hidden == 2:
            state = 2 # Equity Appreciated and Hidden is 2
        if current_hidden == 3:
            state = 3 # Equity Appreciated and Hidden is 3
        if local_sharpe == None:
            "no sharpe ratio"
        elif local_sharpe >= 1:
            state += 4
        else:
            state += 8
    else:
        if current_hidden == 0:
            state = 12 # Equity Appreciated and Hidden is 0
        if current_hidden == 1:
            state = 13 # Equity Appreciated and Hidden is 1
        if current_hidden == 2:
            state = 14 # Equity Appreciated and Hidden is 2
        if current_hidden == 3:
            state = 15 # Equity Appreciated and Hidden is 3

    return state
# Function to find the profit from trades
def determine_payoff(pointer, trade, inPortfolio):
    # Hold the value that the equity was purchased at
    global priceAtPurchase
    if inPortfolio:  # Stock is already owned
        if trade == 0:  # Cannot rebuy the equity; return delta
            print 'Holding Equity at $' + str(round(data['EQUITY'
                    ][pointer], 2))
            print 'Purchase Price: $' + str(round(priceAtPurchase, 2))
            inPortfolio = True
            return (0, inPortfolio)
        if trade == 1:  # Sell the Equity
            inPortfolio = False  # Remove Equity from portfolio
            ret = float(data['EQUITY'][pointer] - priceAtPurchase) / float(priceAtPurchase) + slippage
            print '** Equity sold at $' + str(round(data['EQUITY'
                    ][pointer], 2))
            return (ret, inPortfolio)
    if inPortfolio == False:  # Equity is not owned
        if trade == 0:  # Buy the equity
            inPortfolio = True  # Add it to the portfolio
            print '** Equity bought at $' + str(round(data['EQUITY'
                    ][pointer], 2))  # Display Price Equity was purchased at
            priceAtPurchase = data['EQUITY'][pointer]  # Record the price at which the Equity was purchased
            return (0.0, inPortfolio)
        if trade == 1:  # Sell
            inPortfolio = False
            print 'Out of the market at $' + str(round(data['EQUITY'
                    ][pointer], 2))
            return (0.0, inPortfolio)
 
# aim for profit and stability.
def buildReward(n_periods, trade_prev, trade_cur, ret):
    return ret

# Global variables will be moved into a profit class at next commit
priceAtPurchase = 0
position_ret_curve = []
 
# Runs RL script
def run():
    global position_ret_curve
    # Builds the Q-Table
    q_table = build_q_table(STATES, ACTIONS)
    inPortfolio = False
    # Assuming 0 profit -- or a portfolio with a reference of $0
    returns = []
    hold_returns = []
    trade_periods = []
    # Move through all possible trades
    trade_prev = 1
    n_round_trips = 0
    wins = 0
    losses = 0
    n_periods = 0
    for x in range(1, TOTAL_TRADES - 1):
        # RL Agent chooses the trade
        trade = choose_trade(x, q_table, inPortfolio)
        cur_state = select_state(x, inPortfolio)
        print "cur state:", cur_state
        # Find the payoff from the trade
        ret, inPortfolio = determine_payoff(x, trade, inPortfolio)
        # track signle interval position ret at next
        if inPortfolio:
            position_ret_one_interval = (data['EQUITY'][x + 1] - data['EQUITY'][x]) / float(data['EQUITY'][x])
            position_ret_curve.append(position_ret_one_interval)
        else:
            position_ret_curve = [] # end tracking return curve
        # construct next state
        next_state = select_state(x + 1, inPortfolio)
        # Display to user
        print 'Return from instance: ' + str(ret)
        # Determine trade.
        if trade == 0:
            n_periods += 1
        
        reward = 0
        if ret != 0:
            n_round_trips += 1
            if ret >= 0:
                wins += 1
            else:
                losses += 1
            trade_periods.append(n_periods)
            reward = buildReward(n_periods, trade_prev, trade, ret)
            n_periods = 0
        # build return arrays
        returns.append(ret)
        hold_ret = (data['EQUITY'][x] - data['EQUITY'][x - 1]) / data['EQUITY'][x - 1]
        hold_returns.append(hold_ret)

        trade_prev = trade
        q_predict = q_table.iloc[cur_state, trade]
        # If statement for last trade, tweak this
        if x == TOTAL_TRADES-1:
            q_target = reward + float(agent.gamma) * q_table.iloc[next_state, :
                    ].max()
        else:
            q_target = reward + float(agent.gamma) * q_table.iloc[next_state, :
                    ].max()
        # Append to located cell in Q-Table || Tweak this
        q_table.iloc[cur_state, trade] += float(agent.alpha) * (q_target
                - q_predict)
        print '\n'
    if inPortfolio:
        print "**** Please note that Equity is still held and may be traded later, this may affect profits ****"
    # Return the Q-Table and profit as a tuple
    cum_returns = np.cumprod(np.array(returns) + 1)
    cum_hold_returns = np.cumprod(np.array(hold_returns) + 1)
    win_rate = float(wins) / float(wins + losses)
    hold_only_win_rate = sum(np.array(hold_returns) >= 0) / float(len(hold_returns))
    average_periods = np.mean(trade_periods)
    return (q_table, cum_returns, cum_hold_returns, n_round_trips, average_periods, win_rate, hold_only_win_rate)

# Ensures everything is loaded
if __name__ == '__main__':
    q_table, cum_returns, cum_hold_returns, n_round_trips, average_periods, win_rate, hold_only_win_rate = run()
    print '''\r
Q-table:
'''
    # Add reference column
    q_table["Reference"] = [
        'Hidden is 0 and in portfolio and no local sharpe',
        "Hidden is 1 and in portfolio and no local sharpe",
        "Hidden is 2 and in portfolio and no local sharpe",
        "Hidden is 3 and in portfolio and no local sharpe",
        'Hidden is 0 and in portfolio and local sharpe >= 1',
        "Hidden is 1 and in portfolio and local sharpe >= 1",
        "Hidden is 2 and in portfolio and local sharpe >= 1",
        "Hidden is 3 and in portfolio and local sharpe >= 1",
        'Hidden is 0 and in portfolio and local sharpe < 1',
        "Hidden is 1 and in portfolio and local sharpe < 1",
        "Hidden is 2 and in portfolio and local sharpe < 1",
        "Hidden is 3 and in portfolio and local sharpe < 1",
        'Hidden is 0 and not in portfolio',
        "Hidden is 1 and not in portfolio",
        "Hidden is 2 and not in portfolio",
        "Hidden is 3 and not in portfolio",
        ]
    print q_table
    # Show profits
    END_PORTFOLIO_VALUE = cum_returns[-1] * float(STARTING_PORTFOLIO_VALUE)
    END_HOLD_ONLY_VALUE = cum_hold_returns[-1] * float(STARTING_PORTFOLIO_VALUE)
    print '\nFinal portfolio value from trading ' + str(GIVEN_EQUITY) + ' with starting portfolio of $' + str(STARTING_PORTFOLIO_VALUE) + ': $' + str(END_PORTFOLIO_VALUE)
    print '\nFinal hold only value from trading ' + str(GIVEN_EQUITY) + ' with starting portfolio of $' + str(STARTING_PORTFOLIO_VALUE) + ': $' + str(END_HOLD_ONLY_VALUE)
    print "total round trip:", n_round_trips
    print "average trade periods:", average_periods
    print "win rate:", win_rate
    print "hold only win rate", hold_only_win_rate
    print(cum_returns)
    print(cum_hold_returns)
