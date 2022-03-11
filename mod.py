from freqtrade.strategy import IStrategy , merge_informative_pair
import pandas as pd
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import technical.indicators as ftt


def pivots_points(dataframe: pd.DataFrame , timeperiod=1 , levels=4) -> pd.DataFrame:

    data = {}
    low = qtpylib.rolling_mean(
        series=pd.Series(index=dataframe.index , data=dataframe["low"]) , window=timeperiod
    )
    high = qtpylib.rolling_mean(
        series=pd.Series(index=dataframe.index , data=dataframe["high"]) , window=timeperiod
    )

    # Pivot
    data["pivot"] = qtpylib.rolling_mean(series=qtpylib.typical_price(dataframe), window=timeperiod)
    # R1 = PP + 0.382 * (HIGHprev - LOWprev) ... fibonacci
    data["r1"] = data['pivot'] + 0.382 * (high - low)
    data["rS1"] = data['pivot'] + 0.0955 * (high - low)
    # Resistance #2
    # S1 = PP - 0.382 * (HIGHprev - LOWprev) ... fibonacci
    data["s1"] = data["pivot"] - 0.382 * (high - low)
    data["sR1"] = data["pivot"] - 0.1955 * (high - low)
    # Calculate Resistances and Supports >1
    for i in range(2 , levels + 1):
        prev_support = data["s" + str(i - 1)]
        prev_resistance = data["r" + str(i - 1)]
        # Resitance
        data["r" + str(i)] = (data["pivot"] - prev_support) + prev_resistance
        # Support
        data["s" + str(i)] = data["pivot"] - (prev_resistance - prev_support)
    return pd.DataFrame(index=dataframe.index , data=data)


class hdGen(IStrategy):
    # Optimal timeframe for the strategy
    timeframe = '5m'

    # generate signals from the 1h timeframe
    informative_timeframe = '1h'
    process_only_new_candles = False

    minimal_roi = {
        "0": 7
    }

    # Stoploss:
    stoploss = -0.03

    plot_config = {
        'main_plot': {
            'close_pr1': {'color': 'brown'},
            'high_pr1': {'color': 'green'},
            'pivot': {'color': 'orange'},
            'r1': {'color': 'red'},
            'ema5': {'color': 'blue'},
            'ema10': {'color': 'pink'},
            'ema60': {'color': 'yellow'},
            'ema200': {'color': 'grey'},
            'rS1': {'color': 'blue'},
	    'sR1': {'color': 'green'},
            's1': {'color': 'black'}
        },
        'subplots': {
        }
    }

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair , self.informative_timeframe)
                             for pair in pairs]
        if self.dp:
            for pair in pairs:
                informative_pairs += [(pair, "1d")]

        return informative_pairs

    def slow_tf_indicators(self , dataframe: DataFrame , metadata: dict) -> DataFrame:

        dataframe1d = self.dp.get_pair_dataframe(
            pair=metadata['pair'], timeframe="1d")

        # Pivots Points
        pp = pivots_points(dataframe1d)
        dataframe['pivot'] = pp['pivot']
        dataframe['r1'] = pp['r1']
        dataframe['s1'] = pp['s1']
        dataframe['rS1'] = pp['rS1']
        dataframe['sR1'] = pp['sR1']

        # Definiamo H e C giorno prima
        dataframe['close_pr1'] = dataframe1d['close']
        dataframe['high_pr1'] = dataframe1d['high']

        dataframe = merge_informative_pair(
            dataframe, dataframe1d, self.timeframe, "1d", ffill=True)

        dataframe['ema5'] = ta.EMA(dataframe, timeperiod=5)
        dataframe['ema10'] = ta.EMA(dataframe, timeperiod=10)
        dataframe['ema60'] = ta.EMA(dataframe, timeperiod=60)
        dataframe['ema220'] = ta.EMA(dataframe, timeperiod=220)

        # Start Trading

        dataframe['pivots_ok'] = (
                (dataframe['ema5'] > dataframe['ema10'])
                &
	        (dataframe['ema5'] > dataframe['high_pr1'])
	        &
                (dataframe['ema5'] > dataframe['ema10'])
		&
		(dataframe['volume'] > dataframe['volume'].shift(2))
	).astype('int')

        # # Stiamo Salendo
        # (dataframe['close_pr1'] > dataframe['pivot']) &
        # #(dataframe['close'] > dataframe['ema220']) &
        # (dataframe['ema60'] > dataframe['ema220']) &
        # #(dataframe['pivot'] > dataframe['ema60']) & #Per ora lasciamo ?!
        # # (dataframe['ema60'] > dataframe['pivot']) &
        # # Catch The Pump!
        # #(dataframe['ema10'] > dataframe['ema60']) &
        #
        # # (dataframe['open'] > dataframe['close_pr1'])
        # # |
        # #(dataframe['close'] >= dataframe['pivot'])
        # #&
        # #(dataframe['ema5'] > dataframe['ema10']) &
        # # qtpylib.crossed_above(dataframe['ema5'] , dataframe['close_pr1'])
        # # |
        # qtpylib.crossed_above(dataframe['ema10'], dataframe['rS1'])
        # #(dataframe['ema10'] >= dataframe['rS1'])
 
        dataframe['trending_over'] = (
                (
                    (dataframe['ema10'] > dataframe['ema5'])
                )
                |
                (
                    (dataframe['ema10'] = dataframe['ema5'])
                )#Time for some protections... Btc Or Fake BReak!
                |
                (
                    qtpylib.crossed_above(dataframe['ema60'], dataframe['ema5'])
                )
                |
                (
                    qtpylib.crossed_above(dataframe['ema60'], dataframe['ema10']) #Need some test...
                )
        ).astype('int')

        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe = self.slow_tf_indicators(dataframe, metadata)

        return dataframe

    def populate_buy_trend(self , dataframe: DataFrame , metadata: dict) -> DataFrame:

        dataframe.loc[
            (
                (dataframe['pivots_ok'] > 0)
		|
		(dataframe['close_ok'] > 0)
            ), 'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame , metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['trending_over'] > 0)
            ), 'sell'] = 1
        return dataframe
