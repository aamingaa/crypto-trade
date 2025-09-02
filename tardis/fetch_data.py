API="TD.sDyJS7YZ6oPWSgy2.-vZySO46Lv8avKO.ixQvOq9xdhxqnzC.p1rlPahcqt4F3pp.uORrUOeq0hqYOhV.w6s4"
import nest_asyncio
nest_asyncio.apply()
from tardis_dev import datasets, get_exchange_details
import logging
import pandas as pd
import nest_asyncio
nest_asyncio.apply()
# optionally enable debug logs
# logging.basicConfig(level=logging.DEBUG)

# exchange = 'binance'  #okex，binance等
# exchange_details = get_exchange_details(exchange)


# # pip install tardis-dev
# # requires Python >=3.6
# from tardis_dev import datasets, get_exchange_details
# import logging

# # comment out to disable debug logs
# logging.basicConfig(level=logging.DEBUG)

# # function used by default if not provided via options
# def default_file_name(exchange, data_type, date, symbol, format):
#     return f"{exchange}_{data_type}_{date.strftime('%Y-%m-%d')}_{symbol}.{format}.gz"


# # customized get filename function - saves data in nested directory structure
# def file_name_nested(exchange, data_type, date, symbol, format):
#     return f"{exchange}/{data_type}/{date.strftime('%Y-%m-%d')}_{symbol}.{format}.gz"


# # returns data available at https://api.tardis.dev/v1/exchanges/deribit
# # deribit_details = get_exchange_details("deribit")
# # print(deribit_details)

# def aggregate_to_second_level(df):
#     """
#     将100ms级别的订单簿数据聚合成秒级数据
    
#     参数:
#         df: pandas DataFrame，包含原始100ms级数据，需包含'time'列及各档位数据列
        
#     返回:
#         pandas DataFrame，秒级聚合后的订单簿数据
#     """
#     # 复制原始数据，避免修改原数据
#     df_copy = df.copy()
    
#     # 确保时间列是datetime类型（假设原始time是毫秒级时间戳）
#     # if not pd.api.types.is_datetime64_any_dtype(df_copy['time']):
#     #     df_copy['time'] = pd.to_datetime(df_copy['time'], unit='ms')
    
#     # 将时间列设为索引，便于重采样
#     df_copy = df_copy.set_index('time')
    
#     # 按秒重采样，取每秒最后一个有效数据点作为该秒的代表值
#     # 这反映了每秒结束时的盘口状态
#     second_level_df = df_copy.resample('1S').last()
    
#     # 重置索引，将时间从索引变回列
#     second_level_df = second_level_df.reset_index()
    
#     # 重命名时间列为'time'，保持与原始数据一致
#     second_level_df = second_level_df.rename(columns={'index': 'time'})
    
#     # 移除可能的全NaN行（如果某一秒没有任何数据）
#     second_level_df = second_level_df.dropna(how='all')
    
#     return second_level_df





df = pd.read_csv('/Volumes/Ext-Disk/data/futures/um/tardis/orderbook/ETHUSDT/binance_book_snapshot_5_2019-12-01_ETHUSDT.csv.gz')
df['time'] = pd.to_datetime(df['timestamp'], unit='us')
df.set_index('time', inplace=True)
selected_columns = ['symbol','asks[0].price', 'asks[0].amount', 'bids[0].price', 'bids[0].amount', 'asks[1].price', 'asks[1].amount', 'bids[1].price', 'bids[1].amount', 'asks[2].price', 'asks[2].amount', 'bids[2].price', 'bids[2].amount', 'asks[3].price', 'asks[3].amount', 'bids[3].price', 'bids[3].amount', 'asks[4].price', 'asks[4].amount', 'bids[4].price', 'bids[4].amount']

new_df = df.loc[:, selected_columns]
new_df.columns = ['symbol','ap0', 'av0', 'bp0', 'bv0', 'ap1', 'av1', 'bp1', 'bv1', 'ap2', 'av2', 'bp2', 'bv2', 'ap3', 'av3', 'bp3', 'bv3', 'ap4', 'av4', 'bp4', 'bv4']


print(new_df.head(100))








# datasets.download(
#     # one of https://api.tardis.dev/v1/exchanges with supportsDatasets:true - use 'id' value
#     exchange="binance-futures",
#     # accepted data types - 'datasets.symbols[].dataTypes' field in https://api.tardis.dev/v1/exchanges/deribit,
#     # or get those values from 'deribit_details["datasets"]["symbols][]["dataTypes"] dict above
#     data_types=["book_snapshot_5"],
#     # change date ranges as needed to fetch full month or year for example
#     from_date="2019-12-01",
#     # to date is non inclusive
#     to_date="2019-12-02",
#     # accepted values: 'datasets.symbols[].id' field in https://api.tardis.dev/v1/exchanges/deribit
#     symbols=["ETHUSDT",],
#     # symbol_id = '',
#     # (optional) your API key to get access to non sample data as well
#     api_key=API,
#     # (optional) path where data will be downloaded into, default dir is './datasets'
#     download_dir="/Users/aming/project/python/crypto-trade/tardis/dataset",
#     # (optional) - one can customize downloaded file name/path (flat dir strucure, or nested etc) - by default function 'default_file_name' is used
#     # get_filename=default_file_name,
#     # (optional) file_name_nested will download data to nested directory structure (split by exchange and data type)
#     # get_filename=file_name_nested,
# )


# iterate over and download all data for every symbol
# for symbol in exchange_details["datasets"]["symbols"]:
#     # alternatively specify datatypes explicitly ['trades', 'incremental_book_L2', 'quotes'] etc
#     # see available options https://docs.tardis.dev/downloadable-csv-files#data-types
#     data_types = symbol["dataTypes"]
#     if data_types not in ['book_snapshot_5']:
#         continue

#     symbol_id = symbol["id"]
#     from_date =  symbol["availableSince"]
#     to_date = symbol["availableTo"]

#     # skip groupped symbols
#     # if symbol_id in ['PERPETUALS', 'SPOT', 'FUTURES']:
#     if symbol_id not in ['PERPETUALS']:
#         continue

#     print(f"Downloading {exchange} {data_types} for {symbol_id} from {from_date} to {to_date}")

#     # each CSV dataset format is documented at https://docs.tardis.dev/downloadable-csv-files#data-types
#     # see https://docs.tardis.dev/downloadable-csv-files#download-via-client-libraries for full options docs
#     datasets.download(
#         exchange = exchange,
#         data_types = data_types,
#         from_date =  from_date,
#         to_date = to_date,
#         symbols = [symbol_id],
#         # TODO set your API key here
#         api_key = API,
#         # path where CSV data will be downloaded into
#         download_dir = "./datasets4",
#     )