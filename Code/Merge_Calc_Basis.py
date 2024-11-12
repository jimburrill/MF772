import pandas as pd
import numpy as np


bond_df = pd.read_excel("MF772/Data/ExploratoryData/5YR_Rates.xlsx")
cds_df = pd.read_excel("MF772/Data/ExploratoryData/5YR_CDS_Spreads.xlsx")
Tbills_df = pd.read_excel("MF772/Data/ExploratoryData/T-Bills.xlsx")
bond_df['Date'] = pd.to_datetime(bond_df['Date'])
cds_df['Date'] = pd.to_datetime(cds_df['Date'])
Tbills_df['Date'] = pd.to_datetime(Tbills_df['Date'])


bond_df.iloc[:,1:] = bond_df.iloc[:,1:]/100
cds_df.iloc[:,1:] = cds_df.iloc[:,1:]/10000
Tbills_df.iloc[:,1:] = Tbills_df.iloc[:,1:]/100

def calculate_basis(country):
    merged_df = pd.merge(cds_df[['Date', f'{country} 5Y CDS']], bond_df[['Date', f'{country} 5Y Yield']], on="Date", how="inner")
    merged_df = pd.merge(merged_df, Tbills_df[['Date', f'{country} RF']], on="Date", how="inner")
    merged_df.dropna(inplace=True)
    merged_df[f'{country} Basis'] = merged_df[f'{country} 5Y Yield'] - (merged_df[f'{country} 5Y CDS'])
    merged_df[f'{country} (Basis - RF)'] = merged_df[f'{country} Basis'] - merged_df[f'{country} RF']
    return merged_df

US = calculate_basis("US")
UK = calculate_basis("UK")
Germany = calculate_basis("Germany")
Japan = calculate_basis("Japan")
China = calculate_basis("China")

Combined_Countries = US
Combined_Countries = pd.merge(Combined_Countries, UK, on='Date', how='inner')
Combined_Countries = pd.merge(Combined_Countries, Germany, on='Date', how='inner')
Combined_Countries = pd.merge(Combined_Countries, Japan, on='Date', how='inner')
Combined_Countries = pd.merge(Combined_Countries, China, on='Date', how='inner')

print(Combined_Countries)
