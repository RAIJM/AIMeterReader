import pandas as pd

df = pd.read_excel('manual_label.xlsx')

print(df.head())

for i in range(len(df)):
   df['File Name'][i] = df['File Name'][i][:-4]

df.to_csv('manual_lablel_final.csv')
