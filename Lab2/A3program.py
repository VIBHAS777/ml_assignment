import statistics
import matplotlib.pyplot as plt
import pandas as pd

def load_data():
    df = pd.read_excel("Lab Session Data.xlsx", sheet_name="IRCTC Stock Price")
    return df

def price_stats(df):
    prices = df.iloc[:, 3]
    mean_price = statistics.mean(prices)
    var_price = statistics.variance(prices)
    print("mean of price", mean_price)
    print("variance in prices are:", var_price)
    return prices

def mean_price_on_wednesdays(df):
    prices_on_wed = df[df.iloc[:, 2] == "Wed"].iloc[:, 3]
    mean_wed = statistics.mean(prices_on_wed) if not prices_on_wed.empty else None
    print("mean price on wednesdays are", mean_wed)

def mean_price_in_april(df):
    prices_on_apr = df[df.iloc[:, 1] == "Apr"].iloc[:, 3]
    mean_apr = statistics.mean(prices_on_apr) if not prices_on_apr.empty else None
    print("mean price on april are", mean_apr)

def prob_loss(df, chg):
    chg_loss = df[chg < 0]
    prob = len(chg_loss) / len(chg)
    print("probability of making a loss over the stock is ", prob)

def prob_profit_on_wed(df, chg):
    profit_on_wed = df[(df.iloc[:, 2] == "Wed") & (df.iloc[:, 8] > 0)]
    prob = len(profit_on_wed) / len(chg)
    print("probability of making profit on wednesday", prob)
    return len(profit_on_wed)

def prob_profit_given_wed(df, profit_on_wed_count):
    num_wed = len(df[df.iloc[:, 2] == "Wed"])
    prob = profit_on_wed_count / num_wed
    print("probability of making profit given it is wednesday", prob)

def plot_day_vs_chg(df, chg):
    plt.figure(figsize=(12, 8))
    plt.scatter(df.iloc[:, 2], chg)
    plt.xlabel("Day")
    plt.ylabel("Chg%")
    plt.title("Chg% vs Day")
    plt.show()

def main():
    df = load_data()
    prices = price_stats(df)
    mean_price_on_wednesdays(df)
    print("mean population price is ", statistics.mean(prices))
    mean_price_in_april(df)
    print("mean population price is ", statistics.mean(prices))
    chg = df.iloc[:, 8]
    prob_loss(df, chg)
    profit_on_wed_count = prob_profit_on_wed(df, chg)
    prob_profit_given_wed(df, profit_on_wed_count)
    plot_day_vs_chg(df, chg)

main()
