import os
import sys
import itertools
from tqdm import tqdm
from time import sleep

import warnings
warnings.filterwarnings('ignore')
#import pprint

path = os.getcwd()
parent_path = os.path.abspath(os.path.join(path, os.pardir))
data_path = str(parent_path) + "/holt_winters/data_processing"
model_path = str(parent_path) + "/holt_winters/modeling"

sys.path.append(data_path)
sys.path.append(model_path)


from data_handler import get_data, get_stores, get_families, get_time_series, get_splitted_df, fill_values, structure_predictions, save_predictions
from holt_winters import holt_winters_forecast

forecast_days = 30


def run_forecast():
        print("Getting and transforming data")
        df = get_data()
        stores = get_stores(df)
        families = ['GROCERY I']#, 'BEVERAGES', 'PRODUCE', 'CLEANING','DAIRY'] # get_families(df)

        print("Iterating Forecast")
        #iterations = len(itertools.product(stores, families))
        for i in tqdm(range(len(stores))):
                for s in stores:
                        for f in families:
                                df_info = get_splitted_df(df, f, s)
                                df_ts = get_time_series(df_info)
                                df_ts = fill_values(df_ts)
                                if not df_ts.empty and df_ts['sales'].sum() > 0:
                                        df_yhat = holt_winters_forecast(df_ts, forecast_days)
                                        df_pred = structure_predictions(df['date'].max(), df_yhat, f, s)
                                        save_predictions(df['date'].max(), df_pred)
                                else:
                                        pass
                sleep(0.02)
        
        return print("Forecast finished")

run_forecast()




        





