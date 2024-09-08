import pandas as pd
from datetime import datetime

# load the inference csv files
csv1_file = '../infer_csv/cage_reproduced_VA_inference.csv'
csv2_file = '../infer_csv/nofusion_VA_inference.csv'

df1 = pd.read_csv(csv1_file)
df2 = pd.read_csv(csv2_file)

mode = ["maxfusion", "minfusion", "meanfusion", "weightedfusion"] 
current_time = datetime.now().strftime("%Y%m%d")

for m in mode: 
    if m == "maxfusion":
        output_file = f'../infer_csv/latefusion_max_VA_inference_{current_time}.csv'
        if 'val_pred' in df1.columns and 'val_pred' in df2.columns and 'aro_pred' in df1.columns and 'aro_pred' in df2.columns:
            # Create a new DataFrame 
            df_max = df1.copy()
            df_max['val_pred'] = df1['val_pred'].combine(df2['val_pred'], max)
            df_max['aro_pred'] = df1['aro_pred'].combine(df2['aro_pred'], max)
            df_max.to_csv(output_file, index=False)
            print(f'Max val_pred and aro_pred calculated and saved to {output_file}.')
        else:
            print('One of the CSV files does not contain the required columns (val_pred or aro_pred).')


    if m == "minfusion" :
        output_file = f'../infer_csv/latefusion_min_VA_inference_{current_time}.csv'
        if 'val_pred' in df1.columns and 'val_pred' in df2.columns and 'aro_pred' in df1.columns and 'aro_pred' in df2.columns:
            # Create a new DataFrame 
            df_min = df1.copy()
            df_min['val_pred'] = df1['val_pred'].combine(df2['val_pred'], min)
            df_min['aro_pred'] = df1['aro_pred'].combine(df2['aro_pred'], min)
            df_min.to_csv(output_file, index=False)
            print(f'Mean val_pred and aro_pred calculated and saved to {output_file}.')
        else:
            print('One of the CSV files does not contain the required columns (val_pred or aro_pred).')


    if m == "meanfusion" :
        output_file = f'../infer_csv/latefusion_mean_VA_inference_{current_time}.csv'
        if 'val_pred' in df1.columns and 'val_pred' in df2.columns and 'aro_pred' in df1.columns and 'aro_pred' in df2.columns:
            # Create a new DataFrame 
            df_mean = df1.copy()
            df_mean['val_pred'] = (df1['val_pred'] + df2['val_pred']) / 2
            df_mean['aro_pred'] = (df1['aro_pred'] + df2['aro_pred']) / 2
            df_mean.to_csv(output_file, index=False)
            print(f'Mean val_pred and aro_pred calculated and saved to {output_file}.')
        else:
            print('One of the CSV files does not contain the required columns (val_pred or aro_pred).')

    if m == "weightedfusion" :
        output_file = f'../infer_csv/latefusion_weighted_VA_inference_{current_time}.csv'
        if 'val_pred' in df1.columns and 'val_pred' in df2.columns and 'aro_pred' in df1.columns and 'aro_pred' in df2.columns:
            # Creat a weighted fusion DataFrame
            df_weighted = df1.copy()
            weight1 = 0.6
            weight2 = 0.4
            df_weighted['val_pred'] = df1['val_pred'] * weight1 + df2['val_pred'] * weight2
            df_weighted['aro_pred'] = df1['aro_pred'] * weight1 + df2['aro_pred'] * weight2
            df_weighted.to_csv(output_file, index=False)
            print(f'Weighted mean val_pred and aro_pred calculated and saved to {output_file}.')
        else:
            print('One of the CSV files does not contain the required columns (val_pred or aro_pred).')