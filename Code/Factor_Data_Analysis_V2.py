# Jim Burrill

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns


########################## Functional Definitions ####################################
#%%
# Function to perform regression and return the results
def perform_regression(y, x):
    # Add constant to independent variable (for intercept), adds a column of 1s, x^0
    x_with_const = sm.add_constant(x)
    
    # Fit the OLS regression model
    model = sm.OLS(y, x_with_const)
    results = model.fit()

    return results, x_with_const


# Function to store regression results
def store_regression_results(results, country, factor_name):
    
    # Store relevant data (coefficients, p-values, R-squared)
    row = {
        "model_number": country,
        "Factor": factor_name,
        "r_squared": results.rsquared,
        "adj_r_squared": results.rsquared_adj
    }
    
    for k, (coef, pval) in enumerate(zip(results.params, results.pvalues)):
        row[f"Beta{k}"] = coef
        row[f"pval{k}"] = pval
    
    return row


def store_regression_results_multi_factor(results, country, factor_names, number_factors):
    # Store general model information (R-squared, adjusted R-squared)
    row = {
        "model_number": country,
        "num_factors": num_factors,  # Track the pass when the factors were added
        "r_squared": results.rsquared,
        "adj_r_squared": results.rsquared_adj
    }
    
    # Store coefficients and p-values for each factor
    for k, factor_name in enumerate(factor_names):
        row[f"Beta_{factor_name}"] = results.params[k + 1]  # params[0] is intercept, so start from 1
        row[f"pval_{factor_name}"] = results.pvalues[k + 1]
    
    # Store intercept separately (params[0] is the intercept)
    row["Intercept"] = results.params[0]
    row["Intercept_pval"] = results.pvalues[0]
    
    return row


def dictionary_to_excel(file_name, export_data):
    # Write dictionary sub-dataframs into individual excel sheets
    with pd.ExcelWriter(file_name, engine='xlsxwriter') as writer:
        for sheet, data_dict in export_data.items():
            df = pd.DataFrame(data_dict)
            df.to_excel(writer, sheet_name=sheet, index=False)  # Write each DataFrame to a different sheet



#%%

########################## Read in Data ####################################
#%%
# Define the countries of interest
countries = ['US', 'UK', 'Germany', 'Japan', 'China']


# Read in the factor time series data, Extract column names for each set of data
gdp_gr = pd.read_excel('Factor_Master_Data.xlsx', sheet_name='GDP GR (%)')
names_gdp = gdp_gr.columns
gdp_gr.set_index('Date', inplace=True)
gdp_gr = gdp_gr / 100           # percent to decimal

budget_gdp = pd.read_excel('Factor_Master_Data.xlsx', sheet_name='Budget GDP (%)')
names_budget = budget_gdp.columns
budget_gdp.set_index('Date', inplace=True)
budget_gdp = budget_gdp / 100           # percent to decimal

macro_uncertainty = pd.read_excel('Factor_Master_Data.xlsx', sheet_name='Macro Uncertainty TR')  #No need to do pct change here
names_macro = macro_uncertainty.columns
macro_uncertainty.set_index('Date', inplace=True)

financial_uncertainty = pd.read_excel('Factor_Master_Data.xlsx', sheet_name='Financial Uncertainty TR')  #No need to do pct change here
names_financial = financial_uncertainty.columns
financial_uncertainty.set_index('Date', inplace=True)       # Too correlated with Macro Uncertainty, leave out

inflation = pd.read_excel('Factor_Master_Data.xlsx', sheet_name='Inflation (%)')
names_inflation = inflation.columns
inflation.set_index('Date', inplace=True)
inflation = inflation / 100           # percent to decimal

one_yr_tr = pd.read_excel('Factor_Master_Data.xlsx', sheet_name='1yr TR')          # Using as a proxy for Rf given no data for China
names_1yr = one_yr_tr.columns
one_yr_tr.set_index('Date', inplace=True)
one_yr_tr = 1 + (one_yr_tr / 100)
one_yr_tr = one_yr_tr.pct_change()

index_tr = pd.read_excel('Factor_Master_Data.xlsx', sheet_name='Index TR')
names_index = index_tr.columns
index_tr.set_index('Date', inplace=True)
index_tr = 1 + (index_tr / 100)
index_tr = index_tr.pct_change()

term_structure_tr = pd.read_excel('Factor_Master_Data.xlsx', sheet_name='Term Structure TR')
names_term = term_structure_tr.columns
term_structure_tr.set_index('Date', inplace=True)
term_structure_tr = 1 + (term_structure_tr / 100)
term_structure_tr = term_structure_tr.pct_change()

one_yr_tr = pd.read_excel('Factor_Master_Data.xlsx', sheet_name='1yr TR')          # Using as a proxy for Rf given no data for China
names_1yr = one_yr_tr.columns
one_yr_tr.set_index('Date', inplace=True)
one_yr_tr = 1 + (one_yr_tr / 100)
one_yr_tr = one_yr_tr.pct_change()


bid_ask = pd.read_excel('Factor_Master_Data.xlsx', sheet_name='Bid Ask (%)')
names_bidask = bid_ask.columns
bid_ask.set_index('Date', inplace=True)
bid_ask = bid_ask.pct_change()

forex = pd.read_excel('Factor_Master_Data.xlsx', sheet_name='Forex')  #No need to do pct change here
names_forex = forex.columns
forex.set_index('Date', inplace=True)


# Read in the country CDS-Bond Basis and Basis - Rf time series data, will be Y variable in regressions
all_cds_data_ts = pd.read_csv('Temp_Merged_Countries_Basis.csv')
all_cds_data_ts['Date'] = pd.to_datetime(all_cds_data_ts['Date'])
all_cds_data_ts = all_cds_data_ts.sort_values(by='Date', ascending=True)

cds_ts = all_cds_data_ts[['Date', 'US 5Y CDS', 'UK 5Y CDS', 'Germany 5Y CDS', 'Japan 5Y CDS', 'China 5Y CDS']]
names_cds = cds_ts.columns
cds_ts.set_index('Date', inplace=True)

basis_ts = all_cds_data_ts[['Date', 'US Basis', 'UK Basis', 'Germany Basis', 'Japan Basis', 'China Basis']]
names_basis = basis_ts.columns
basis_ts.set_index('Date', inplace=True)

basis_rf_ts = all_cds_data_ts[['Date', 'US (Basis - RF)', 'UK (Basis - RF)', 'Germany (Basis - RF)', 'Japan (Basis - RF)',
                               'China (Basis - RF)']]
names_basis_rf = basis_rf_ts.columns
basis_rf_ts.set_index('Date', inplace=True)





#%%


#################### Factor Correlation Analysis #################################### 
#%%
# Look at correlation structure of the time series for each country
# Account for the three differnet possibilities for macro uncertainty (settled on 3 month ahead)
# Ensure to calculate index EXCESS return over 1yr rate (proxy for Rf)
corr_matrices = []
for i in range(1, len(names_gdp)):
    excess_index_return = index_tr[names_index[i]] - one_yr_tr[names_1yr[i]]
    '''country_ts = pd.DataFrame({names_gdp[i]: gdp_gr[names_gdp[i]], names_budget[i]: budget_gdp[names_budget[i]], 
                        names_macro[2]: macro_uncertainty[names_macro[2]], names_inflation[i]: inflation[names_inflation[i]], 
                        names_index[i]: excess_index_return, names_term[i]: term_structure_tr[names_term[i]],  
                        names_1yr[i]: one_yr_tr[names_1yr[i]], names_bidask[i]: bid_ask[names_bidask[i]], 
                        names_forex[i]: forex[names_forex[i]]})'''
    country_ts = pd.DataFrame({names_gdp[i]: gdp_gr[names_gdp[i]], names_budget[i]: budget_gdp[names_budget[i]],
                        names_macro[2]: macro_uncertainty[names_macro[2]], names_inflation[i]: inflation[names_inflation[i]], 
                        names_index[i]: excess_index_return, names_term[i]: term_structure_tr[names_term[i]],  
                        names_bidask[i]: bid_ask[names_bidask[i]], names_forex[i]: forex[names_forex[i]]})
    country_ts = country_ts.dropna()
    corr_matrix = country_ts.corr()
    corr_matrices.append(corr_matrix)
    
    # Plot the correlation matrix heatmap
    plt.figure(figsize=(8, 6))  # Set the figure size
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5, fmt='.2f')
    plt.title(f'{countries[i-1]}: Correlation Matrix Heatmap')




#%%

########################## Factor Time Series Analysis ####################################
#%%
for i in range(1, len(names_gdp)):
    excess_index_return = index_tr[names_index[i]] - one_yr_tr[names_1yr[i]]
    '''country_ts = pd.DataFrame({'Date': gdp_gr.index.to_numpy(), names_gdp[i]: gdp_gr[names_gdp[i]], names_budget[i]: budget_gdp[names_budget[i]], 
                        names_macro[2]: macro_uncertainty[names_macro[2]], names_inflation[i]: inflation[names_inflation[i]], 
                        names_index[i]: excess_index_return, names_term[i]: term_structure_tr[names_term[i]],  
                        names_1yr[i]: one_yr_tr[names_1yr[i]], names_bidask[i]: bid_ask[names_bidask[i]], 
                        names_forex[i]: forex[names_forex[i]]})'''
    country_ts = pd.DataFrame({'Date': gdp_gr.index.to_numpy(), names_gdp[i]: gdp_gr[names_gdp[i]], names_budget[i]: budget_gdp[names_budget[i]], 
                        names_macro[2]: macro_uncertainty[names_macro[2]], names_inflation[i]: inflation[names_inflation[i]], 
                        names_index[i]: excess_index_return, names_term[i]: term_structure_tr[names_term[i]],  
                        names_bidask[i]: bid_ask[names_bidask[i]], names_forex[i]: forex[names_forex[i]]})
    country_ts = country_ts.dropna()
    country_ts.set_index('Date', inplace=True)
    
    names_country_ts = country_ts.columns
    num_factors = len(names_country_ts)
    
    # Create subplots (grid layout)
    rows = (num_factors + 1) // 2  # Dynamic rows, adjust to number of factors
    fig, axs = plt.subplots(rows, 2, figsize=(15, rows * 5))  # 2 columns
    axs = axs.flatten()  # Flatten to 1D array for easy iteration
    
    # Plot each factor in a separate subplot
    for j, column in enumerate(country_ts.columns):
        axs[j].plot(country_ts.index, country_ts[column], label=column)
        axs[j].set_title(column)
        axs[j].grid(True)
        axs[j].legend()
    
    # Remove empty subplots (if any)
    if num_factors % 2 != 0:
        fig.delaxes(axs[-1])
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Show the full figure for the current country
    plt.suptitle(f"Factor Time Series for {countries[i-1]}", y=1.02, fontsize=16)
    plt.show()




#%%


#################### CDS time series analysis ####################################  
#%% 
regression_results = []
for i in range(1, len(names_gdp)):
    excess_index_return = index_tr[names_index[i]] - one_yr_tr[names_1yr[i]]
    '''country_ts = pd.DataFrame({names_cds[i]: cds_ts[names_cds[i]], names_gdp[i]: gdp_gr[names_gdp[i]], 
                               names_budget[i]: budget_gdp[names_budget[i]], names_macro[2]: macro_uncertainty[names_macro[2]], 
                               names_inflation[i]: inflation[names_inflation[i]], names_index[i]: excess_index_return, 
                               names_term[i]: term_structure_tr[names_term[i]], names_1yr[i]: one_yr_tr[names_1yr[i]], 
                               names_bidask[i]: bid_ask[names_bidask[i]], names_forex[i]: forex[names_forex[i]]})'''
    country_ts = pd.DataFrame({names_cds[i]: cds_ts[names_cds[i]], names_gdp[i]: gdp_gr[names_gdp[i]], 
                               names_budget[i]: budget_gdp[names_budget[i]], names_macro[2]: macro_uncertainty[names_macro[2]], 
                               names_inflation[i]: inflation[names_inflation[i]], names_index[i]: excess_index_return, 
                               names_term[i]: term_structure_tr[names_term[i]], 
                               names_bidask[i]: bid_ask[names_bidask[i]], names_forex[i]: forex[names_forex[i]]})
    country_ts = country_ts.dropna()
    
    names_country_ts = country_ts.columns
    num_factors = len(names_country_ts) - 1  # Exclude the dependent variable
    y = country_ts[names_country_ts[0]]     # Choose the y variable
    
    # Create subplots (grid layout)
    rows = (num_factors + 1) // 2  # Dynamic rows, adjust to number of factors
    fig, axs = plt.subplots(rows, 2, figsize=(15, rows * 5))  # 2 columns
    axs = axs.flatten()  # Flatten to 1D array for easy iteration
    
    # loop over each factor one at a time
    for j in range(1, len(names_country_ts)):
        x = country_ts[names_country_ts[j]]
        
        results, x_with_const = perform_regression(y, x)
        
        transformed_results = store_regression_results(results, countries[i-1], names_country_ts[j])
        
        regression_results.append(transformed_results)
        
        # Scatter plot of the actual data in the current subplot
        sns.scatterplot(x=x, y=y, ax=axs[j - 1], color='blue', label='Data')
        
        # Predicted values (regression line)
        y_pred = results.predict(x_with_const)
        
        # Plot the regression line
        axs[j - 1].plot(x, y_pred, color='red', label='Regression Line')
        
        # Add labels and title for each subplot
        axs[j - 1].set_title(f"{names_country_ts[0]} vs {names_country_ts[j]}")
        axs[j - 1].set_xlabel(names_country_ts[j])  # X-axis: factor name
        axs[j - 1].set_ylabel(names_country_ts[0])  # Y-axis: CDS spread or dependent variable

        # Add legend
        axs[j - 1].legend()
        
    # Hide unused subplots if num_factors is odd
    if num_factors % 2 != 0:
        axs[-1].axis('off')  # Hide the last subplot if not needed
        
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Show the full figure for the current country
    plt.suptitle(f"Factor Regressions for {countries[i-1]}", y=1.02, fontsize=16)
    plt.show()
        
# Create DataFrame from all regression results
cds_analysis_single_factor = pd.DataFrame(regression_results)
cds_analysis_single_factor.to_excel('cds_single_factor_analysis.xlsx', engine='xlsxwriter', index=False)

  

#%%


#################### Basis time series analysis #################################### 
#%%  
regression_results = []
for i in range(1, len(names_gdp)):
    excess_index_return = index_tr[names_index[i]] - one_yr_tr[names_1yr[i]]
    '''country_ts = pd.DataFrame({names_basis[i]: basis_ts[names_basis[i]], names_gdp[i]: gdp_gr[names_gdp[i]], 
                               names_budget[i]: budget_gdp[names_budget[i]], names_macro[2]: macro_uncertainty[names_macro[2]], 
                               names_inflation[i]: inflation[names_inflation[i]], names_index[i]: excess_index_return, 
                               names_term[i]: term_structure_tr[names_term[i]], names_1yr[i]: one_yr_tr[names_1yr[i]], 
                               names_bidask[i]: bid_ask[names_bidask[i]], names_forex[i]: forex[names_forex[i]]})'''
    country_ts = pd.DataFrame({names_basis[i]: basis_ts[names_basis[i]], names_gdp[i]: gdp_gr[names_gdp[i]], 
                               names_budget[i]: budget_gdp[names_budget[i]], names_macro[2]: macro_uncertainty[names_macro[2]], 
                               names_inflation[i]: inflation[names_inflation[i]], names_index[i]: excess_index_return, 
                               names_term[i]: term_structure_tr[names_term[i]], 
                               names_bidask[i]: bid_ask[names_bidask[i]], names_forex[i]: forex[names_forex[i]]})
    country_ts = country_ts.dropna()
    
    names_country_ts = country_ts.columns
    num_factors = len(names_country_ts) - 1  # Exclude the dependent variable
    y = country_ts[names_country_ts[0]]      # Choose the y variable
    
    # Create subplots (grid layout)
    rows = (num_factors + 1) // 2  # Dynamic rows, adjust to number of factors
    fig, axs = plt.subplots(rows, 2, figsize=(15, rows * 5))  # 2 columns
    axs = axs.flatten()  # Flatten to 1D array for easy iteration
    
    # loop over each factor one at a time
    for j in range(1, len(names_country_ts)):
        x = country_ts[names_country_ts[j]]
        
        results, x_with_const = perform_regression(y, x)
        
        transformed_results = store_regression_results(results, countries[i-1], names_country_ts[j])
        
        regression_results.append(transformed_results)
        
        # Scatter plot of the actual data in the current subplot
        sns.scatterplot(x=x, y=y, ax=axs[j - 1], color='blue', label='Data')
        
        # Predicted values (regression line)
        y_pred = results.predict(x_with_const)
        
        # Plot the regression line
        axs[j - 1].plot(x, y_pred, color='red', label='Regression Line')
        
        # Add labels and title for each subplot
        axs[j - 1].set_title(f"{names_country_ts[0]} vs {names_country_ts[j]}")
        axs[j - 1].set_xlabel(names_country_ts[j])  # X-axis: factor name
        axs[j - 1].set_ylabel(names_country_ts[0])  # Y-axis: CDS spread or dependent variable

        # Add legend
        axs[j - 1].legend()
        
    # Hide unused subplots if num_factors is odd
    if num_factors % 2 != 0:
        axs[-1].axis('off')  # Hide the last subplot if not needed
        
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Show the full figure for the current country
    plt.suptitle(f"Factor Regressions for {countries[i-1]}", y=1.02, fontsize=16)
    plt.show()
        
# Create DataFrame from all regression results
basis_analysis_single_factor = pd.DataFrame(regression_results)
basis_analysis_single_factor.to_excel('basis_single_factor_analysis.xlsx', engine='xlsxwriter', index=False)




#%%


#################### Basis-Rf time series analysis #################################### 
#%%  
regression_results = []
for i in range(1, len(names_gdp)):
    excess_index_return = index_tr[names_index[i]] - one_yr_tr[names_1yr[i]]
    '''country_ts = pd.DataFrame({names_basis_rf[i]: basis_rf_ts[names_basis_rf[i]], names_gdp[i]: gdp_gr[names_gdp[i]], 
                               names_budget[i]: budget_gdp[names_budget[i]], names_macro[2]: macro_uncertainty[names_macro[2]], 
                               names_inflation[i]: inflation[names_inflation[i]], names_index[i]: excess_index_return, 
                               names_term[i]: term_structure_tr[names_term[i]], names_1yr[i]: one_yr_tr[names_1yr[i]], 
                               names_bidask[i]: bid_ask[names_bidask[i]], names_forex[i]: forex[names_forex[i]]})'''
    country_ts = pd.DataFrame({names_basis_rf[i]: basis_rf_ts[names_basis_rf[i]], names_gdp[i]: gdp_gr[names_gdp[i]], 
                               names_budget[i]: budget_gdp[names_budget[i]], names_macro[2]: macro_uncertainty[names_macro[2]], 
                               names_inflation[i]: inflation[names_inflation[i]], names_index[i]: excess_index_return, 
                               names_term[i]: term_structure_tr[names_term[i]], 
                               names_bidask[i]: bid_ask[names_bidask[i]], names_forex[i]: forex[names_forex[i]]})
    country_ts = country_ts.dropna()
    
    names_country_ts = country_ts.columns
    num_factors = len(names_country_ts) - 1  # Exclude the dependent variable
    y = country_ts[names_country_ts[0]]      # Choose the y variable
    
    # Create subplots (grid layout)
    rows = (num_factors + 1) // 2  # Dynamic rows, adjust to number of factors
    fig, axs = plt.subplots(rows, 2, figsize=(15, rows * 5))  # 2 columns
    axs = axs.flatten()  # Flatten to 1D array for easy iteration
    
    # loop over each factor one at a time
    for j in range(1, len(names_country_ts)):
        x = country_ts[names_country_ts[j]]
        
        results, x_with_const = perform_regression(y, x)
        
        transformed_results = store_regression_results(results, countries[i-1], names_country_ts[j])
        
        regression_results.append(transformed_results)
        
        # Scatter plot of the actual data in the current subplot
        sns.scatterplot(x=x, y=y, ax=axs[j - 1], color='blue', label='Data')
        
        # Predicted values (regression line)
        y_pred = results.predict(x_with_const)
        
        # Plot the regression line
        axs[j - 1].plot(x, y_pred, color='red', label='Regression Line')
        
        # Add labels and title for each subplot
        axs[j - 1].set_title(f"{names_country_ts[0]} vs {names_country_ts[j]}")
        axs[j - 1].set_xlabel(names_country_ts[j])  # X-axis: factor name
        axs[j - 1].set_ylabel(names_country_ts[0])  # Y-axis: CDS spread or dependent variable

        # Add legend
        axs[j - 1].legend()
        
    # Hide unused subplots if num_factors is odd
    if num_factors % 2 != 0:
        axs[-1].axis('off')  # Hide the last subplot if not needed
        
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Show the full figure for the current country
    plt.suptitle(f"Factor Regressions for {countries[i-1]}", y=1.02, fontsize=16)
    plt.show()
        
# Create DataFrame from all regression results
basis_rf_analysis_single_factor = pd.DataFrame(regression_results)
basis_rf_analysis_single_factor.to_excel('basis_rf_single_factor_analysis.xlsx', engine='xlsxwriter', index=False)





#%%


#################### CDS Multifactor Analysis #################################### 
#%% 
cds_analysis_multi_dict = {}
for i in range(1, len(names_gdp)):
    regression_results = []
    excess_index_return = index_tr[names_index[i]] - one_yr_tr[names_1yr[i]]
    '''country_ts = pd.DataFrame({names_cds[i]: cds_ts[names_cds[i]], names_gdp[i]: gdp_gr[names_gdp[i]], 
                               names_budget[i]: budget_gdp[names_budget[i]], names_macro[2]: macro_uncertainty[names_macro[2]], 
                               names_inflation[i]: inflation[names_inflation[i]], names_index[i]: excess_index_return, 
                               names_term[i]: term_structure_tr[names_term[i]], names_1yr[i]: one_yr_tr[names_1yr[i]], 
                               names_bidask[i]: bid_ask[names_bidask[i]], names_forex[i]: forex[names_forex[i]]})'''
    country_ts = pd.DataFrame({names_cds[i]: cds_ts[names_cds[i]], names_gdp[i]: gdp_gr[names_gdp[i]], 
                               names_budget[i]: budget_gdp[names_budget[i]], names_macro[2]: macro_uncertainty[names_macro[2]], 
                               names_inflation[i]: inflation[names_inflation[i]], names_index[i]: excess_index_return, 
                               names_term[i]: term_structure_tr[names_term[i]], 
                               names_bidask[i]: bid_ask[names_bidask[i]], names_forex[i]: forex[names_forex[i]]})
    country_ts = country_ts.dropna()
    
    names_country_ts = country_ts.columns
    num_factors = len(names_country_ts) - 1  # Exclude the dependent variable
    y = country_ts[names_country_ts[0]]     # Choose the y variable
    
    # loop over each factor one at a time, adding one at a time
    for j in range(2, len(names_country_ts)+1):
        x = country_ts[names_country_ts[1:j]]
        
        results, x_with_const = perform_regression(y, x)
        
        transformed_results = store_regression_results_multi_factor(results, countries[i-1], names_country_ts[1:j], (j-1))
        
        regression_results.append(transformed_results)
        
    # Create DataFrame from all regression results
    cds_analysis_multi_factor = pd.DataFrame(regression_results)
    
    # Store the DataFrame for the current country in the dictionary
    cds_analysis_multi_dict[countries[i-1]] = cds_analysis_multi_factor

# Export to excel    
dictionary_to_excel('cds_multi_factor_analysis.xlsx', cds_analysis_multi_dict)



#%%


#################### Basis Multifactor Analysis #################################### 
#%% 
basis_analysis_multi_dict = {}
for i in range(1, len(names_gdp)):
    regression_results = []
    excess_index_return = index_tr[names_index[i]] - one_yr_tr[names_1yr[i]]
    '''country_ts = pd.DataFrame({names_basis[i]: basis_ts[names_basis[i]], names_gdp[i]: gdp_gr[names_gdp[i]], 
                               names_budget[i]: budget_gdp[names_budget[i]], names_macro[2]: macro_uncertainty[names_macro[2]], 
                               names_inflation[i]: inflation[names_inflation[i]], names_index[i]: excess_index_return, 
                               names_term[i]: term_structure_tr[names_term[i]], names_1yr[i]: one_yr_tr[names_1yr[i]], 
                               names_bidask[i]: bid_ask[names_bidask[i]], names_forex[i]: forex[names_forex[i]]})'''
    country_ts = pd.DataFrame({names_basis[i]: basis_ts[names_basis[i]], names_gdp[i]: gdp_gr[names_gdp[i]], 
                               names_budget[i]: budget_gdp[names_budget[i]], names_macro[2]: macro_uncertainty[names_macro[2]], 
                               names_inflation[i]: inflation[names_inflation[i]], names_index[i]: excess_index_return, 
                               names_term[i]: term_structure_tr[names_term[i]], 
                               names_bidask[i]: bid_ask[names_bidask[i]], names_forex[i]: forex[names_forex[i]]})
    country_ts = country_ts.dropna()
    
    names_country_ts = country_ts.columns
    num_factors = len(names_country_ts) - 1  # Exclude the dependent variable
    y = country_ts[names_country_ts[0]]     # Choose the y variable
    
    # loop over each factor one at a time, adding one at a time
    for j in range(2, len(names_country_ts)+1):
        x = country_ts[names_country_ts[1:j]]
        
        results, x_with_const = perform_regression(y, x)
        
        transformed_results = store_regression_results_multi_factor(results, countries[i-1], names_country_ts[1:j], (j-1))
        
        regression_results.append(transformed_results)
        
    # Create DataFrame from all regression results
    basis_analysis_multi_factor = pd.DataFrame(regression_results)
    
    # Store the DataFrame for the current country in the dictionary
    basis_analysis_multi_dict[countries[i-1]] = basis_analysis_multi_factor
    
# Export to excel    
dictionary_to_excel('basis_multi_factor_analysis.xlsx', basis_analysis_multi_dict)

    

#%%


#################### Basis-Rf Multifactor Analysis #################################### 
#%% 
basis_rf_analysis_multi_dict = {}
for i in range(1, len(names_gdp)):
    regression_results = []
    excess_index_return = index_tr[names_index[i]] - one_yr_tr[names_1yr[i]]
    '''country_ts = pd.DataFrame({names_basis_rf[i]: basis_rf_ts[names_basis_rf[i]], names_gdp[i]: gdp_gr[names_gdp[i]], 
                               names_budget[i]: budget_gdp[names_budget[i]], names_macro[2]: macro_uncertainty[names_macro[2]], 
                               names_inflation[i]: inflation[names_inflation[i]], names_index[i]: excess_index_return, 
                               names_term[i]: term_structure_tr[names_term[i]], names_1yr[i]: one_yr_tr[names_1yr[i]], 
                               names_bidask[i]: bid_ask[names_bidask[i]], names_forex[i]: forex[names_forex[i]]})'''
    country_ts = pd.DataFrame({names_basis_rf[i]: basis_rf_ts[names_basis_rf[i]], names_gdp[i]: gdp_gr[names_gdp[i]], 
                               names_budget[i]: budget_gdp[names_budget[i]], names_macro[2]: macro_uncertainty[names_macro[2]], 
                               names_inflation[i]: inflation[names_inflation[i]], names_index[i]: excess_index_return, 
                               names_term[i]: term_structure_tr[names_term[i]], 
                               names_bidask[i]: bid_ask[names_bidask[i]], names_forex[i]: forex[names_forex[i]]})
    country_ts = country_ts.dropna()
    
    names_country_ts = country_ts.columns
    num_factors = len(names_country_ts) - 1  # Exclude the dependent variable
    y = country_ts[names_country_ts[0]]     # Choose the y variable
    
    # loop over each factor one at a time, adding one at a time
    for j in range(2, len(names_country_ts)+1):
        x = country_ts[names_country_ts[1:j]]
        
        results, x_with_const = perform_regression(y, x)
        
        transformed_results = store_regression_results_multi_factor(results, countries[i-1], names_country_ts[1:j], (j-1))
        
        regression_results.append(transformed_results)
        
    # Create DataFrame from all regression results
    basis_rf_analysis_multi_factor = pd.DataFrame(regression_results)
    
    # Store the DataFrame for the current country in the dictionary
    basis_rf_analysis_multi_dict[countries[i-1]] = basis_rf_analysis_multi_factor
    
# Export to excel    
dictionary_to_excel('basis_rf_multi_factor_analysis.xlsx', basis_rf_analysis_multi_dict)




#%%



#################### Ridge Regression: Basis-Rf Multifactor Analysis #################################### 
# When almost any lasso is intriduced all factors go to zero
#%% 
basis_rf_analysis_elastic_dict = {}
for i in range(1, len(names_gdp)):
    regression_results = []
    excess_index_return = index_tr[names_index[i]] - one_yr_tr[names_1yr[i]]
    '''country_ts = pd.DataFrame({names_basis_rf[i]: basis_rf_ts[names_basis_rf[i]], names_gdp[i]: gdp_gr[names_gdp[i]], 
                               names_budget[i]: budget_gdp[names_budget[i]], names_macro[2]: macro_uncertainty[names_macro[2]], 
                               names_inflation[i]: inflation[names_inflation[i]], names_index[i]: excess_index_return, 
                               names_term[i]: term_structure_tr[names_term[i]], names_1yr[i]: one_yr_tr[names_1yr[i]], 
                               names_bidask[i]: bid_ask[names_bidask[i]], names_forex[i]: forex[names_forex[i]]})'''
    country_ts = pd.DataFrame({names_basis_rf[i]: basis_rf_ts[names_basis_rf[i]], names_gdp[i]: gdp_gr[names_gdp[i]], 
                               names_budget[i]: budget_gdp[names_budget[i]], names_macro[2]: macro_uncertainty[names_macro[2]], 
                               names_inflation[i]: inflation[names_inflation[i]], names_index[i]: excess_index_return, 
                               names_term[i]: term_structure_tr[names_term[i]], 
                               names_bidask[i]: bid_ask[names_bidask[i]], names_forex[i]: forex[names_forex[i]]})
    country_ts = country_ts.dropna()
    
    names_country_ts = country_ts.columns
    num_factors = len(names_country_ts) - 1  # Exclude the dependent variable
    y = country_ts[names_country_ts[0]]     # Choose the y variable
    
    # Do the multifactor ridge regression
    for j in range(2, len(names_country_ts)+1):
        # Scale the data for ridge regression
        scaler = StandardScaler()
        x = country_ts[names_country_ts[1:j]]
        x = scaler.fit_transform(x)
        
        # L1_ratio controls the mix of lasso and ridge), higher ratio more lass0  
        # Higher values of alpha imply stronger regularization for ridge portion
        elastic_net_model = ElasticNet(alpha=1, l1_ratio=0)
        elastic_net_model.fit(x, y)
        elastic_net_betas = elastic_net_model.coef_
                
        regression_results.append(elastic_net_betas)
        
    # Create DataFrame from all regression results
    basis_rf_analysis_elastic = pd.DataFrame(regression_results)
    
    # Store the DataFrame for the current country in the dictionary
    basis_rf_analysis_elastic_dict[countries[i-1]] = basis_rf_analysis_elastic
    
# Export to excel    
dictionary_to_excel('basis_rf_multi_factor_elastic.xlsx', basis_rf_analysis_elastic_dict)


