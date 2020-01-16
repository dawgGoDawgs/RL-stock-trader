# Reinforcement Learning Equity Trader
# By Matija Krolo || github.com/Kroat
# To use:
# 1.) Ensure all necessary libraries are installed on your machine (Numpy, Scipy, Pandas)
# 2.) Place RL-Trader.py in a project directory
# 3.) cd into the project directory
# 4.) Pass the command:
#       RL-Trader.py [EQUITY] [START DATE - DAY/MONTH/YEAR] [STARTING PORTFOLIO VALUE] [HOW MANY TRADES TO RUN BEFORE REINFORCEMENT LEARNING BEGINS]
#
# ex: RL-Trader.py F 1/1/2000 1000 200
# ^ This runs RL-Trader against Ford's historical data and analyzes 200 trades before the Reinforcement Learning begins with a beginning portfolio of $1,000
# ** Please note that this RL script only trades one equity at a time!

# Edit these values to change how the RL brain learns
EPSILON = .3
ALPHA = .1
GAMMA = .1

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
EQUITY = web.get_data_yahoo(GIVEN_EQUITY, end='1/1/2019', start=START_DATE,
                            interval='d')
MKT_VOLATIILTY = web.get_data_yahoo('^VIX', end='1/1/2019',
                                    start=START_DATE, interval='d')
RF_Rate = web.get_data_yahoo('^TNX', end='1/1/2019', start=START_DATE,
                             interval='d')
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

print EQUITY.shape
print pct.shape
print adx.shape
print rsi.shape
print beta.shape
print correl.shape
print mfi.shape

# Pack diff and volume for training.
X = np.column_stack([pct, adx, rsi, beta, correl, mfi])
col_mean = np.nanmean(X, axis=0)
inds = np.where(np.isnan(X))
X[inds] = np.take(col_mean, inds[1])
# Make an HMM instance and execute fit
num_components = 3
model = GaussianHMM(n_components=num_components, covariance_type="diag", n_iter=1000).fit(X_train)
hidden_states = model.predict(X)

# Why not edit this?
STATES = 8
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
def choose_trade(pointer, q_table):
    # Logic is only running
    if pointer < int(TRADES_TO_RUN):
        print ("Reinforcement Learning not initiated yet, Q-Table still building.")
    # Find the trade decision from our trade logic
    analytic_decision = state_logic(pointer, data)
    # Select state from Q-Table
    state_actions = q_table.iloc[select_state(pointer), :]
    # If the greedy factor is less than a randomly distributed number, if there are no values
    # on the Q-table, or if less than half the possible trades have been run without our trading logic,
    # return our analytical trade logic decision
    if np.random.uniform() > float(agent.epsilon) or state_actions.all() == 0 or pointer < int(TRADES_TO_RUN):
        return analytic_decision
    # Otherwise, return what has been working
    else:
        maximum = state_actions.idxmax()
        if str(maximum) == 'buy':
            return 0
        if str(maximum) == 'sell':
            return 1

# Selects the state on the Q-Table
def select_state(pointer):
    # Find the current price of the equity
    current_price = data['EQUITY'][pointer]
    # Find the previous price of the equity
    previous_price = data['EQUITY'][pointer - 1]
    # Find the current market volatility
    current_vol = data['SIGMA'][pointer]
    # Find the previous market volatility
    previous_vol = data['SIGMA'][pointer - 1]
    # Find the current RF rate
    current_rf = data['RF'][pointer] if not np.isnan(data['RF'][pointer]) else 1.0
    # Find the previous RF rate
    previous_rf = data['RF'][pointer - 1] if not np.isnan(data['RF'][pointer - 1]) else 1.0

    if current_price > previous_price:
        if current_vol > previous_vol:
            if current_rf > previous_rf:
                return 0  # Equity Appreciated and Market Vol Increase and Risk Free Rate Increase
            if current_rf <= previous_rf:
                return 1  # Equity Appreciated and Market Vol Increase and Risk Free Rate Decrease
        if current_vol <= previous_vol:
            if current_rf > previous_rf:
                return 2  # Equity Appreciated and Market Vol Decrease and Risk Free Rate Increase
            if current_rf <= previous_rf:
                return 3  # Equity Appreciated and Market Vol Decrease and Risk Free Rate Decrease
    if current_price <= previous_price:
        if current_vol > previous_vol:
            if current_rf > previous_rf:
                return 4  # Equity Deppreciated and Market Vol Increase and Risk Free Rate Increase
            if current_rf <= previous_rf:
                return 5  # Equity Deppreciated and Market Vol Increase and Risk Free Rate Decrease
        if current_vol <= previous_vol:
            if current_rf > previous_rf:
                return 6  # Equity Deppreciated and Market Vol Decrease and Risk Free Rate Increase
            if current_rf <= previous_rf:
                return 7  # Equity Deppreciated and Market Vol Decrease and Risk Free Rate Decrease

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
            print '** Equity sold at $' + str(round(data['EQUITY'
                    ][pointer], 2))
            return (data['EQUITY'][pointer] - priceAtPurchase, inPortfolio)
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
 

# Global variables will be moved into a profit class at next commit
priceAtPurchase = 0
 
# Runs RL script
def run():
    # Builds the Q-Table
    q_table = build_q_table(STATES, ACTIONS)
    inPortfolio = False
    aggregate_profit = []
    # Assuming 0 profit -- or a portfolio with a reference of $0
    profit = 0
    # Move through all possible trades
    for x in range(TOTAL_TRADES):
        # RL Agent chooses the trade
        trade = choose_trade(x - 1, q_table)
        # Find the payoff from the trade
        result, inPortfolio = determine_payoff(x, trade, inPortfolio)
        # Display to user
        print 'Profit from instance: ' + str(round(result, 2))
        # Append result from trade to aggregate profit
        aggregate_profit.append(result)
        # Slows down the script
        time.sleep(.05)
        # Append to profit
        profit += result
        q_predict = q_table.iloc[select_state(x), trade]
        # If statement for last trade, tweak this
        if x == TOTAL_TRADES-1:
            q_target = result + float(agent.gamma) * q_table.iloc[select_state(x), :
                    ].max()
        else:
            q_target = result + float(agent.gamma) * q_table.iloc[select_state(x), :
                    ].max()
        # Append to located cell in Q-Table || Tweak this
        q_table.iloc[select_state(x), trade] += float(agent.alpha) * (q_target
                - q_predict)
        print '\n'
    if inPortfolio:
        print "**** Please note that Equity is still held and may be traded later, this may affect profits ****"
    # Return the Q-Table and profit as a tuple
    profit = np.sum(aggregate_profit)
    return (q_table, profit)

# Ensures everything is loaded
if __name__ == '__main__':
    q_table, profit = run()
    print '''\r
Q-table:
'''
    # Add reference column
    q_table["Reference"] = [
        'When Equity Appreciated and Market Vol Increase and RF rate Increase',
        'When Equity Appreciated and Market Vol Increase and RF rate Decrease',
        'When Equity Appreciated and Market Vol Decrease and RF rate Increase',
        'When Equity Appreciated and Market Vol Decrease and RF rate Decrease',
        'When Equity Depreciated and Market Vol Increase and RF rate Increase',
        'When Equity Depreciated and Market Vol Increase and RF rate Decrease',
        'When Equity Depreciated and Market Vol Decrease and RF rate Increase',
        'When Equity Depreciated and Market Vol Decrease and RF rate Decrease'
        ]
    print q_table
    # Show profits
    calc_profits = 1 + round(profit, 2)/100.0
    calc_profits = calc_profits * float(STARTING_PORTFOLIO_VALUE)
    print '\nProfits from trading ' + str(GIVEN_EQUITY) + ' with starting portfolio of $' + str(STARTING_PORTFOLIO_VALUE) + ': $' + str(calc_profits)
