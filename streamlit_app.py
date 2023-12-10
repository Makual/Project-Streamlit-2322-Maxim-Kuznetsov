
import streamlit as st

def main():
    st.title("Streamlit App converted from Jupyter Notebook")
    
    # Streamlit code can go here. For now, let's just run the code from the notebook.
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('CalomirisPritchett_data.csv')
df

#Select useful columns

df = df[['Sales Date', 'Sex', 'Age', 'Color', 'Price']]
df.head(15)

df['Date'] = pd.to_datetime(df['Sales Date'], format='%m/%d/%Y', errors='coerce') #if wrong date format, delete the row
df = df.sort_values(by='Date')
df

df['Price'].replace('.', pd.NA, inplace=True)
df['Sex'].replace('.', pd.NA, inplace=True)
df['Color'].replace('.', pd.NA, inplace=True)
df['Sales Date'].replace('.', pd.NA, inplace=True)

df = df.dropna(subset=['Price', 'Sex', 'Age', 'Date'])
df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
df

df.info()

df.describe()

df['Year'] = df['Date'].dt.year

# Count the number of records for each year
yearly_counts = df['Year'].value_counts().sort_index()

# Create a bar graph
plt.bar(yearly_counts.index, yearly_counts.values, color='skyblue')
plt.xlabel('Year')
plt.ylabel('Number of Records')
plt.title('Number of Records per Year')
plt.show()

#we should analyse data from 1856-1865
yearly_counts = dict(df['Year'].value_counts().sort_index())
yearly_counts = pd.Series({i: yearly_counts[i] for i in yearly_counts if int(i) >= 1856 and int(i) <= 1865})

# Create a bar graph
plt.bar(yearly_counts.index, yearly_counts.values, color='skyblue')
plt.xlabel('Year')
plt.ylabel('Number of Records')
plt.title('Number of Records per Year')
plt.show()

plt.figure(figsize=(8, 6))
plt.hist(df[df['Price'] < 5000]['Price'], bins=10, color='skyblue') # To avoid data outliers
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.title('Price Distribution')
plt.show()

df['Price'].describe()

df['Price'].median()

plt.figure(figsize=(8, 6))
plt.hist(df['Age'], bins=10, edgecolor='k', color='skyblue') # To avoid data outliers
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Age Distribution')
plt.show()

df['Age'].describe()

df['Age'].median()

plt.figure(figsize=(5, 5))
plt.hist(df[df['Sex'].isin(['F', 'M'])]['Sex'], bins=2, edgecolor='k', color='skyblue') # To avoid data outliers
plt.xlabel('Sex')
plt.ylabel('Frequency')
plt.title('Sex Distribution')
plt.show()

color_counts = df['Color'].value_counts()

total_count = len(df)
percentage_threshold = 1
color_counts = color_counts[color_counts / total_count * 100 >= percentage_threshold]

plt.figure(figsize=(8, 8))
plt.pie(color_counts, labels=color_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('Color Distribution')
plt.axis('equal')
plt.show()

df['Sex_Code'] = df['Sex'].map({'F': 0, 'M': 1})
df = df[df['Sex'].isin(['F', 'M'])]

df['Color_Code'] = df['Color'].map({'Negro': 0, 'Mulatto': 1,'Griff': 2, 'Black': 3})
df = df[df['Color'].isin(['Negro', 'Mulatto', 'Griff', 'Black'])]
df

#annual statistics

cut_df = df[(df['Year'] >= 1856) & (df['Year'] <= 1861)]


annual_stat = {}
for year in range(1856, 1862):
    annual_stat[year] = cut_df[cut_df['Year'] == year].describe()

plt.bar(range(1856, 1862), [annual_stat[i]['Price']['mean'] for i in annual_stat], color='skyblue')
plt.xlabel('Year')
plt.ylabel('Price')
plt.title('Annual mean price')
plt.show()

plt.bar(range(1856, 1862), [annual_stat[i]['Age']['mean'] for i in annual_stat], color='skyblue')
plt.xlabel('Year')
plt.ylabel('Age')
plt.title('Annual mean age')
plt.show()

#skin color statistics
color_stat = {}

for color in ['Negro', 'Mulatto', 'Griff', 'Black']:
    color_group = df[df['Color'] == color]
    if not color_group.empty:
        color_stat[color] = color_group.describe()

color_stat = pd.DataFrame({color: [color_stat[color]['Price']['mean'], color_stat[color]['Age']['mean']] for color in ['Negro', 'Mulatto', 'Griff', 'Black']})
color_stat = color_stat.rename(index={0: 'Price', 1: 'Age'})

color_stat

male_df = df[df['Sex'] == 'M']
female_df = df[df['Sex'] == 'F']

male_df

female_df

age_bins = range(0, int(female_df['Age'].max()) + 5, 5)


female_df['Age_Group'] = pd.cut(female_df['Age'], bins=age_bins, right=False)


age_group_mean_price = female_df[female_df['Price'] < 1500].groupby('Age_Group')['Price'].mean()


plt.figure(figsize=(10, 6))
age_group_mean_price.plot(kind='bar', color='skyblue')
plt.title('Mean Price of Female Slaves in Each Age Group')
plt.xlabel('Age Group')
plt.ylabel('Mean Price')
plt.xticks(rotation=45)
plt.show()

age_bins = range(0, int(male_df['Age'].max()) + 5, 5)


male_df['Age_Group'] = pd.cut(male_df['Age'], bins=age_bins, right=False)


age_group_mean_price = male_df[male_df['Price'] < 5000].groupby('Age_Group')['Price'].mean()


plt.figure(figsize=(10, 6))
age_group_mean_price.plot(kind='bar', color='skyblue')
plt.title('Mean Price of Male Slaves in Each Age Group')
plt.xlabel('Age Group')
plt.ylabel('Mean Price')
plt.xticks(rotation=45)
plt.show()

correlation_female = female_df['Age'].corr(female_df['Price'])
correlation_male = male_df['Age'].corr(male_df['Price'])

print(f'Correlation between Age and Price for Female slaves: {correlation_female}')
print(f'Correlation between Age and Price for Male slaves: {correlation_male}')

male_df_age_0_20 = male_df[(male_df['Age'] >= 0) & (male_df['Age'] <= 20)]
male_df_age_20_80 = male_df[(male_df['Age'] > 20) & (male_df['Age'] <= 80)]

# Calculate the correlation for each subset
male_correlation_0_20 = male_df_age_0_20['Age'].corr(male_df_age_0_20['Price'])
male_correlation_20_80 = male_df_age_20_80['Age'].corr(male_df_age_20_80['Price'])

female_df_age_0_20 = female_df[(female_df['Age'] >= 0) & (female_df['Age'] <= 20)]
female_df_age_20_80 = female_df[(female_df['Age'] > 20) & (female_df['Age'] <= 80)]

# Calculate the correlation for each subset
female_correlation_0_20 = female_df_age_0_20['Age'].corr(female_df_age_0_20['Price'])
female_correlation_20_80 = female_df_age_20_80['Age'].corr(female_df_age_20_80['Price'])

print(f'Correlation between Age (0-20) and Price for Males: {male_correlation_0_20}')
print(f'Correlation between Age (20-80) and Price for Males: {male_correlation_20_80}')

print(f'Correlation between Age (0-20) and Price for Females: {female_correlation_0_20}')
print(f'Correlation between Age (20-80) and Price for Females: {female_correlation_20_80}')

if __name__ == "__main__":
    main()
