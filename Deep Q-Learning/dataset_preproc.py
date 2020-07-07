import pandas as pd

ds_input_path = 'data/EURUSD_Candlestick_1_M_BID_01.01.2017-31.12.2017.csv'
ds_output_path = 'data/EURUSD_Candlestick_1_M_BID_01.01.2017-31.12.2017-FILTERED.csv'
dataset = pd.read_csv(ds_input_path, sep=',', header=0, dtype='str')

# Deleting instances with 0 volumes
ds_filtered = dataset[~dataset['Volume'].isin(['0'])]

# Resetting the index
ds_filtered = ds_filtered.reset_index()

# Deleting unused fields
del ds_filtered['Local time']
del ds_filtered['Volume']
del ds_filtered['index']

# Check for correct dimensions
print(ds_filtered.shape)

# Exporting dataset
ds_filtered.to_csv(ds_output_path, sep=',', index=False)
