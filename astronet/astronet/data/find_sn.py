import pandas as pd

df = pd.read_csv('tces.csv', header=0, index_col=0)

for row in df.iterrows():
    new_dataframe = pd.DataFrame(columns=columns)
    tics = np.loadtxt(bad_list, dtype=int)
    new_dataframe['tic_id'] = tics
    new_dataframe['src'] = 'qlp'
    new_dataframe['toi_id'] = '.01'
    new_dataframe['Sectors'] = int(bad_list.split('-')[1])
    new_dataframe['Disposition'] = 'J'
    bad_tces = pd.concat([bad_tces, new_dataframe], ignore_index=True)