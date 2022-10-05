import os
from functools import reduce
from datetime import datetime
from time import sleep
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import scipy.optimize as solver
import yfinance as yf


# input:


# 元大台灣高股息證券投資信託基金 Yuanta Taiwan Dividend Plus ETF (0056):
target_stocks = ["2382", "2301", "2357", "2609", "3034",
                   "4938", "1102", "2409", "2303", "2454",
                   "1101", "2002", "2603", "2324", "2377",
                   "2347", "2356", "3231", "3702", "2385",
                   "2376", "3036", "3044", "6176", "5522",
                   "2006", "2449", "2014", "2441"]

# US top 25 stocks (excluding BRK.B):
# target_stocks = ["AAPL", "MSFT", "AMZN", "TSLA", "GOOGL",
#                  "GOOG", "NVDA", "META", "UNH",
#                  "JNJ", "JPM", "V", "PG", "XOM",
#                  "HD", "CVX", "MA", "BAC", "ABBV",
#                  "PFE", "AVGO", "COST", "DIS", "KO"]

# TW top 10 stocks:
# target_stocks = ["2330", '2412', '1301', '2882',
#                  '1303', '2886', '2891', '1216', '5880', '2892']

# Crawl stocks and save as csv


def stocks_data_crawler(target_stocks: list, nation="tw"):
    """Crawl history price of Taiwan or stocks from Yahoo Finance
    Input: 
    target_stocks : a list of stock codes
    nation = 'tw' or 'us', dufault='tw' """
    if not os.path.exists('data'):
        os.makedirs('data')

    for stock_id in target_stocks:
        data = pd.DataFrame()
        if nation == "tw":
            data = yf.Ticker(f"{stock_id}.TW")
        else:
            data = yf.Ticker(stock_id)
        df = data.history(period="max")
        file_name = f"./data/{stock_id}.csv"
        df.to_csv(file_name)
        sleep(1.2)


stocks_data_crawler(target_stocks=target_stocks, nation="us")


def efficient_frontier_maker(
        target_stocks: list, start: str, simulations_target=10**5, risk_free=0.0165, investor_expected_return=0.1):
    """Calculate Efficient Frontier and make figure,
    while Minimum Variance Portfolio and Market Portfolio will be pointed out.

    Input:\n
    targer_stocks = the list of stock codes strings (format example ["2330", '2412'])
    start = the start date of simulation (format example:"2012/10/01")
    simulations_target (simulation_times) = int (default: 10**5)
    risk_free = float (default: 0.0165)
    investor_expected_return = float (default: 0.1)
    """
    if not os.path.exists('output'):
        os.makedirs('output')

    # Inner join the "Close" columns by "Date" column of all stocks' file, then save as CSV
        # Note:
        # If any "Close" value == NaN, the value in that date will be skipped

    for index, stock_id in enumerate(target_stocks):
        # Set index name according to first file and extract "Close" value
        if index == 0:
            df = pd.read_csv(
                f'./data/{stock_id}.csv').set_index("Date").loc[:, "Close"].rename(stock_id)
        else:
            temp_df = pd.read_csv(
                f'./data/{stock_id}.csv').set_index("Date").loc[:, "Close"].rename(stock_id)
            df = pd.concat([df, temp_df], axis=1, join="inner")

        # save inner joined file as CSV
    time = datetime.now().strftime("%m%d-%H%M%S")
    file_name = f"./data/data_for_analysing({time}).csv"
    df.to_csv(file_name)
    print("data_for_analysing saved as csv")

    # Simulations

    # set caculation period and read data
    df = pd.read_csv(file_name, index_col=0, parse_dates=True)
    print(f"Raw data starting date: {str(df.first_valid_index())[:10]}")
    df = df.loc[start:, :]
    end_date = df.tail(1).index.item().strftime("%Y/%m/%d")
    total_stocks = len(df.columns)

    # Set %_change as returns:
    returns = df.pct_change()
    returns = returns[1:]
    # returns.head()

    # Calculate covariance and 1-year expected returns, based on average return
    covariance_matrix = returns.cov()
    stocks_expected_return = ((returns.mean()+1) ** 252)-1

    # Set stocks weights, value=1 & Portfolio_return
    stocks_weights = np.array([1/total_stocks, ]*total_stocks)
    return_list = []
    risk_list = []

    for _ in range(simulations_target):

        weight = np.random.rand(total_stocks)
        weight = weight / sum(weight)

        returns = sum(stocks_expected_return * weight)
        risks = np.sqrt(reduce(np.dot, [weight, covariance_matrix, weight.T]))

        return_list.append(returns)
        risk_list.append(risks)

    #  Create Stochastic Simulations Figure
    # fig = plt.figure(figsize=(10, 6))
    # fig.suptitle('Stochastic Simulations', fontsize=18, fontweight='bold')
    # ax = fig.add_subplot()
    # ax.plot(risk_list, return_list, 'o')
    # ax.set_title(f'n={simulations_target}', fontsize=16)
    # fig.savefig('./output/Stochastic_Simulations.png', dpi=300)

    # Minimum Variance Portfolio, MVP
        # Define a function of standard_deviation to calulate risks
    def standard_deviation(weights):
        return np.sqrt(reduce(np.dot, [weights, covariance_matrix, weights.T]))

        # Set Attributions
    x0 = stocks_weights
    bounds = tuple((0, 1) for x in range(total_stocks))
    constraints = [{'type': 'eq', 'fun': lambda x: sum(x) - 1}]

    # Find MVP
    minimize_variance = solver.minimize(
        standard_deviation, x0=x0, constraints=constraints, bounds=bounds)

    mvp_risk = minimize_variance.fun
    mvp_return = sum(minimize_variance.x * stocks_expected_return)
    output_mvp = []
    output_mvp.append(f"{round(mvp_return, 4)*100}%")
    output_mvp.append(f"{round(mvp_risk, 4)*100}%")
    print(f"MVP return:\n {round(mvp_return, 4)*100}%\n")
    print(f"MVP risk:\n {round(mvp_risk, 4)}")

    # Stock Weight of MVP
    output_mvp.append("")
    for i in range(total_stocks):
        weighted = round(minimize_variance.x[i]*100, 2)
        print(f"{df.columns[i]}'s weight: {weighted}%")
        output_mvp.append(f"{weighted}%")
    print("MVP found")

    # Create Efficient Frontier

    x0 = stocks_weights
    bounds = tuple((0, 1) for x in range(total_stocks))
    return_max = round(max(stocks_expected_return), 3)
    return_min = round(min(stocks_expected_return), 3)

    efficient_fronter_return_range = np.arange(
        return_min, return_max, (round((return_max-return_min)/50, 4)))
    efficient_fronter_risk_list = []

    # print(efficient_fronter_return_range)

    for given_return in efficient_fronter_return_range:
        constraints = [{'type': 'eq', 'fun': lambda x: sum(x) - 1},
                       {'type': 'eq', 'fun': lambda x: sum(x * stocks_expected_return) - given_return}]
        efficient_fronter = solver.minimize(
            standard_deviation, x0=x0, constraints=constraints, bounds=bounds)
        efficient_fronter_risk_list.append(efficient_fronter.fun)
    print("Efficient Frontier created")

    # Market Portfolio

    mp = max((np.array(efficient_fronter_return_range)-risk_free) /
             np.array(efficient_fronter_risk_list))

    for index, retrn in enumerate(efficient_fronter_return_range):
        if mp == (retrn-risk_free)/efficient_fronter_risk_list[index]:
            mp_return = retrn
            constraints = [{'type': 'eq', 'fun': lambda x: sum(x) - 1},
                           {'type': 'eq', 'fun': lambda x: sum(x * stocks_expected_return) - retrn}]
            efficient_fronter = solver.minimize(
                standard_deviation, x0=x0, constraints=constraints, bounds=bounds)
            mp_risk = efficient_fronter.fun

    output_mp = []
    output_mp.append(f"{round(mp_return*100,2)}%")
    output_mp.append(f"{round(mp_risk*100,2)}%")

    print(
        f"Market Portfolio:\n return: {round(mp_return*100,2)}% \n risk: {round(mp_risk*100,2)}%")

    # Stock Weights of Market Portfolio

    constraints = [
        {'type': 'eq', 'fun': lambda x: sum(x) - 1},
        {'type': 'eq', 'fun': lambda x: sum(x * stocks_expected_return) - mp_return}]
    mp_fun = solver.minimize(
        standard_deviation, x0=x0, constraints=constraints, bounds=bounds)

    output_mp.append("")
    for i in range(total_stocks):
        weighted = round(mp_fun.x[i]*100, 2)
        output_mp.append(f"{weighted}%")
        print(f"{df.columns[i]}'s weight: {weighted}%")

    # Investor's Expected Risk and Portfolio

        # Stock Weights of Expected Portfolio
    constraints = [
        {'type': 'eq', 'fun': lambda x: sum(x) - 1},
        {'type': 'eq', 'fun': lambda x: sum(x * stocks_expected_return) - investor_expected_return}]
    ex_fun = solver.minimize(
        standard_deviation, x0=x0, constraints=constraints, bounds=bounds)
    investor_expected_risk = ex_fun.fun

    output_ex = []
    output_ex.append(f"{round(investor_expected_return*100,2)}%")
    output_ex.append(f"{round(investor_expected_risk*100,2)}%")
    output_ex.append("")
    print(
        f"Investor's Expected Return:\n return: {round(investor_expected_return*100,2)}%\n {round(investor_expected_risk*100,2)}%")

    for i in range(total_stocks):
        weighted = round(ex_fun.x[i]*100, 2)
        output_ex.append(f"{weighted}%")
        print(f"{df.columns[i]}'s weight: {weighted}%")

        # Save output as CSV
    name = ["Return:", "Risk:", "Weight:"]
    for i in target_stocks:
        name.append(i)
    dic = {"": name,
           "Minimum Variance Portfolio": output_mvp,
           "Market Portfolio": output_mp,
           "Portfolio for Investor's Expected Return": output_ex}
    df = pd.DataFrame(dic)
    df.to_csv(f"./output/Result-{time}.csv")
    print("Portfolio Result created")

    # Create Efficient Fronter Figure

    fig = plt.figure(figsize=(12, 6))
    fig.suptitle('Efficient Frontier', fontsize=22, fontweight='bold')
    fig.subplots_adjust(top=0.8)
    ax = fig.add_subplot()
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=1))
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=1))

    # Stochastic Simulations Points
    ax0 = ax.scatter(risk_list, return_list,
                     c=(np.array(return_list)-risk_free)/np.array(risk_list),
                     marker='o', label="Stochastic Simulations")
    ax.legend()

    # Efficient Frontier Line
    ax.plot(efficient_fronter_risk_list,
            efficient_fronter_return_range, linewidth=1, color='#251f6b', marker='o',
            markerfacecolor='#251f6b', markersize=5, label='Efficient Frontier')
    ax.legend()

    # Minimum Variance Portfolio Point
    ax.plot(mvp_risk, mvp_return, '*', color='r',
            markerfacecolor='#ed1313',  markersize=20, label='Minimum Variance Portfolio')
    ax.annotate(
        f"   Return:{round(mvp_return*100,2)}%, Risk:{round(mvp_risk*100,2)}%",
        (mvp_risk, mvp_return))
    ax.legend()

    # Market Portfolio Point
    ax.plot(mp_risk, mp_return, '*', color='r',
            markerfacecolor='#ffee11',  markersize=20, label='Market Portfolio')
    ax.annotate(
        f"   Return:{round(mp_return*100,2)}%, Risk:{round(mp_risk*100,2)}%", (mp_risk, mp_return))
    ax.legend()

    #    Investor Expected Return Portfolio Point
    ax.plot(investor_expected_risk, investor_expected_return, '*', color='r',
            markerfacecolor='#007ACC',  markersize=20, label='Investor Expected Portfolio')
    ax.annotate(
        f"   Return:{round(investor_expected_return*100,2)}%, Risk:{round(investor_expected_risk*100,2)}%", (investor_expected_risk, investor_expected_return))
    ax.legend()

    # Sharpe Ratio Color Bar
    fig.colorbar(ax0, ax=ax, label='Sharpe Ratio')

    # Figure Format
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_title(
        f'Period of Simulations:\n{start} - {end_date}\nN={simulations_target}', fontsize=12, loc="left")
    ax.set_xlabel('Risk', fontsize=16)
    ax.set_ylabel('Return', fontsize=16)

    plt.savefig(f'./Output/Efficient_Frontier-{time}.png', dpi=300)
    print("Figure created")


efficient_frontier_maker(
    target_stocks=target_stocks_2,
    start="2012/10/01", investor_expected_return=0.2)
