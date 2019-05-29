import numpy as np
import pandas as pd
import quandl
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from ohlc_plot import pandas_candlestick_ohlc
pd.set_option("display.max_columns", 20)


# We will look at stock prices over the past year, starting at January 1, 2016
start = datetime.datetime(2016, 1, 1)
end = datetime.date.today()

# Let's get Apple stock data; Apple's ticker symbol is AAPL
# First argument is the series we want, second is the source ("yahoo"), third is the start date, fourth is the end date
s = "AAPL"
apple = quandl.get("WIKI/" + s, start_date=start, end_date=end)

type(apple)

print(apple.head())

# apple["Adj. Close"].plot(grid = True)
# plt.show()


# pandas_candlestick_ohlc(apple, adj=True, stick="month")

microsoft, google = (quandl.get("WIKI/" + s, start_date=start, end_date=end) for s in ["MSFT", "GOOG"])

stocks = pd.DataFrame({"AAPL": apple["Adj. Close"],
                       "MSFT": microsoft["Adj. Close"],
                       "GOOG": google["Adj. Close"]})

stocks.head()
# stocks.plot(grid=True)

returns = stocks.apply(lambda x: x/x[0])

# returns.plot()


log_returns = stocks.apply(lambda x: np.log(x) - np.log(x.shift(1)))

log_returns.plot()
plt.show()