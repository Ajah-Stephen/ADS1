# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 17:15:58 2023

@author: user
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import stats
from scipy.stats import skew, kurtosis
import seaborn as sns

"""
   Define a function which takes a filename as argument, 
   reads a dataframe in Worldbank format and returns two dataframes:
   one with years as columns and one with countries as columns.
   Do not forget to clean the transposed dataframe.

"""

def read_worldbank_data(filename):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(filename)
    
    # Extract relevant columns
    df_yrs = df[['Country Name', 'Country Code', 'Series Name', 'Series Code',
                   '1990 [YR1990]', '1995 [YR1995]', '2000 [YR2000]',
                   '2005 [YR2005]', '2010 [YR2010]']]
  
   # Pivot DataFrame for year as columns
    df_yrs = df_yrs.melt(id_vars=['Country Name', 'Country Code',
                                  'Series Name', 'Series Code'],
                             var_name='Year', value_name='Value')
    
    # Filter for 'Year' column Dataframe
    df_yrs['Year'] = df_yrs['Year'].str.extract(r'\[YR(\d+)\]').astype(int)

    # Pivot DataFrame for countries as columns
    df_count = df_yrs.pivot_table(index=['Series Name', 'Series Code', 'Year'],
                                        columns='Country Code', values='Value',
                                        aggfunc='first').reset_index()
    
    return df_yrs, df_count

# Analysed Dataframe
df_yrs, df_count = read_worldbank_data('wordads.csv')

# Print the dataframes
print("DataFrame with Years as Columns:")
print(df_yrs.head())

print("\nDataFrame with Countries as Columns:")
print(df_count.head())

"""

 Explore the statistical properties of a few indicators (Series),
 cross-compare between individual countries ('CHN', 'NGA', 'IND', 'USA')
 and produce appropriate summary statistics. Using aggregated data for 
 regions (Country Code) and other categories. Using .describe() method
 to explore your data and two other statistical methods. ('Correlation Matrix
 and Percentages')
 
 """
 
 # Select series and countries
series = ['CO2 emissions (kt)', 'Urban population',
          'Electric power consumption(kWh per capita)',
          'Forest area (% of land area)']
countries = ['CHN', 'NGA', 'IND', 'USA']

# Clean the data for the selected series and countries
clean_data = df_yrs[df_yrs['Series Name'].isin(series) & df_yrs['Country Code']
                    .isin(countries)]

# Apply statistics using describe() method
apply_stats = clean_data.groupby(['Series Name',
                                  'Country Code'])['Value'].describe()

# Apply correlation matrix
corr_matrix = clean_data.pivot_table(index='Country Code',
                                     columns='Series Name', values='Value',
                                     aggfunc='first').corr()

# Apply percentiles
percent = clean_data.groupby(['Series Name',
                              'Country Code'])['Value'].quantile(
                                  [0.25, 0.5, 0.75]).unstack()
                                  
# Print results
print("\nApply Statistics:")
print(apply_stats)

print("\nCorrelation Matrix:")
print(corr_matrix)

print("\nPercentiles:")
print(percent)

# Select series and countries
series = ['CO2 emissions (kt)', 'Urban population',
          'Electric power consumption (kWh per capita)',
          'Forest area (% of land area)']
countries = ['CHN', 'NGA', 'IND', 'USA']

# Clean Dataframe for the selected Series and Countries
clean_data = df_yrs[df_yrs['Series Name'].isin(series) &
                    df_yrs['Country Code'].isin(countries)]

# Apply statistics for countries
stats = clean_data.groupby(['Country Code', 'Series Name'])['Value'].describe()

# Handle missing values
clean_data_pivot = clean_data.pivot_table(index='Country Code',
                                          columns='Series Name',
                                          values='Value', aggfunc='first')
country_corr_matrix = clean_data_pivot.corr()

# Print results
print("\nResult for Statistics by Country:")
print(stats)

print("\nCorrelation Matrix by Country:")
print(country_corr_matrix)

"""


  Explore and understand any correlations (or lack of) between indicators
  (e.g. 'CO2 emissions (kt)', 'Urban population', 'Electric power 
  consumption (kWh per capita)', 'Forest area (% of land area)').
  And how it varies between country, having any
  correlations or trends changed with time?
  
  """
  
# Select Series and Countries
series = ['CO2 emissions (kt)', 'Urban population',
          'Electric power consumption (kWh per capita)',
          'Forest area (% of land area)']
countries = ['CHN', 'NGA', 'IND', 'USA']

# Clean Dataframe for selected Series and Countries
clean_data = df_yrs[df_yrs['Series Name'].isin(series) &
                    df_yrs['Country Code'].isin(countries)]

# Handle missing values
clean_data_pivot = clean_data.pivot_table(index='Country Code',
                                          columns='Series Name',
                                          values='Value', aggfunc='first')
country_corr_matrix = clean_data_pivot.corr()

# Creating the plot for Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(country_corr_matrix, annot=True, cmap='coolwarm',
            fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix of Series for Selected Countries')
plt.show()

# Select Series and Countries
series = ['CO2 emissions (kt)', 'Urban population',
          'Electric power consumption (kWh per capita)',
          'Forest area (% of land area)']
countries = ['CHN', 'NGA']

# Clean Dataframe for selected Series and Countries
clean_data = df_yrs[df_yrs['Series Name'].isin(series) &
                    df_yrs['Country Code'].isin(countries)]

# Pivot Dataframe for correlation calculation
clean_data_pivot = clean_data.pivot_table(index='Country Code',
                                          columns='Series Name',
                                          values='Value', aggfunc='first')

# Calculate the correlation between CHN and NGA
corr_chn_nga = clean_data_pivot.loc['CHN'].corr(clean_data_pivot.loc['NGA'])

# Print the result correlation
print("Correlation between CHN and NGA for each series:")
print(corr_chn_nga)

# Clean the dataframe Series and Countries
clean_data = df_yrs[df_yrs['Series Name'].isin(series) &
                    df_yrs['Country Code'].isin(countries)]

# Pivotting the Dataframe for Correlation calculation
clean_data_pivot = clean_data.pivot_table(index='Country Code',
                                          columns='Series Name',
                                          values='Value', aggfunc='first')

# Calculate Correlation between CHN and NGA
corr_chn_nga = clean_data_pivot.loc['CHN'].corr(clean_data_pivot.loc['NGA'])

# Convert the correlation values to a DataFrame for plotting
corr_df = pd.DataFrame({'Correlation': corr_chn_nga}, index=series)

# Plotting the correlation between CHN and NGA for Sries
plt.figure(figsize=(12, 10))
bar_plot = sns.barplot(x=corr_df.index, y='Correlation', data=corr_df,
                       palette='RdBu_r')

# Adding labels to Series CHN and NGA
for index, value in enumerate(corr_df['Correlation']):
    label_text = f'{value:.2f}\n(CHN)' if index == 0 else f'{value:.2f}\n(NGA)'
    bar_plot.text(index, value, label_text, ha='center', va='bottom')

plt.title('Correlation between CHN and NGA for Selected Series')
plt.xlabel('Series')
plt.ylabel('Correlation Coefficient')
plt.show()

# Select Series, Countries, and Years
series = ['CO2 emissions (kt)', 'Urban population',
          'Electric power consumption (kWh per capita)',
          'Forest area (% of land area)']
countries = ['CHN', 'NGA', 'IND', 'USA']
years = [1990, 1995, 2000, 2005, 2010]

# Clean Dataframe for the selected Series, Countries, and Years
clean_data = df_yrs[df_yrs['Series Name'].isin(series) &
                    df_yrs['Country Code'].isin(countries) &
                    df_yrs['Year'].isin(years)]

# Apply the plot size
plt.figure(figsize=(12, 8))

# Apply the plot for each Series for the countries over the years
for indict in series:
    for country in countries:
        data_to_plot = clean_data[(clean_data['Series Name'] == indict) &
                                  (clean_data['Country Code'] == country)]
        plt.plot(data_to_plot['Year'], data_to_plot['Value'],
                 label=f'{indict} - {country}')
        
        plt.title('Trend of Series for Countries Over Years')
plt.xlabel('Year')
plt.ylabel('Value')
plt.legend()
plt.show()

# Select indicators, countries, and years
indicators = ['CO2 emissions (kt)', 'Urban population',
              'Electric power consumption (kWh per capita)',
              'Forest area (% of land area)']
countries = ['CHN', 'NGA', 'IND', 'USA']
years = [1990, 1995, 2000, 2005, 2010]

# Clean Dataframe for the selected Series, Countries, and Years
clean_data = df_yrs[df_yrs['Series Name'].isin(series) &
                    df_yrs['Country Code'].isin(countries) &
                    df_yrs['Year'].isin(years)]

# Create box plots for each indicator
plt.figure(figsize=(12, 8))
for i, indict in enumerate(series, 1):
    plt.subplot(2, 2, i)
    sns.boxplot(x='Series Name', y='Value',
                data=clean_data[clean_data['Series Name'] == indict])
    plt.title(f'Box Plot of {indict}')
    
    plt.tight_layout()
plt.show()

















































    
    

    
   
   
   


    
    
    
    







