
def main():
    st.title("Streamlit App from Jupyter Notebook")

import streamlit as st


st.markdown(    """# **New Orlean's Slave Sales analysis**""")


st.markdown(    """##In this notebook Ill analyse a dataset of 15,377 slave sales from 1856 - 1861""")


st.markdown(    """# Importing libraries""")


with st.echo():
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np


st.markdown(    """# Data opening""")


with st.echo():
    df = pd.read_csv('CalomirisPritchett_data.csv')
    df


st.markdown(    """#Data description
##Sales Date - Sales date of transaction (MM/DD/YYYY format). This is NOT the date the sale was recorded in conveyance office.
##Sex Male (M) or female (F). - Gender often inferred from sale record, such as Negro vs. Negress, mulatto vs. mulattress, etc.
##Age - Age in years
##Color - Description of slave’s skin color, but may also indicate ancestry (such as mulatto)
##Price - Price associated with slaves. For group sales with a single price, only one price is listed for first slave, and for other slaves, entry is blank""")


with st.echo():
    #Select useful columns
    
    df = df[['Sales Date', 'Sex', 'Age', 'Color', 'Price']]
    df.head(15)


st.markdown(    """#Sorting data by date""")


with st.echo():
    df['Date'] = pd.to_datetime(df['Sales Date'], format='%m/%d/%Y', errors='coerce') #if wrong date format, delete the row
    df = df.sort_values(by='Date')
    df


st.markdown(    """#Data cleanup
The key data is Price, so we should delete rows with NaN values. The same with the Age and Sex and Date
. is the NaN too

""")


with st.echo():
    df['Price'].replace('.', pd.NA, inplace=True)
    df['Sex'].replace('.', pd.NA, inplace=True)
    df['Color'].replace('.', pd.NA, inplace=True)
    df['Sales Date'].replace('.', pd.NA, inplace=True)
    
    df = df.dropna(subset=['Price', 'Sex', 'Age', 'Date'])
    df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
    df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
    df


st.markdown(    """#Data Overview""")


with st.echo():
    df.info()


with st.echo():
    df.describe()


st.markdown(    """##Date analysis""")


with st.echo():
    df['Year'] = df['Date'].dt.year
    
    # Count the number of records for each year
    yearly_counts = df['Year'].value_counts().sort_index()
    
    # Create a bar graph
    plt.bar(yearly_counts.index, yearly_counts.values, color='skyblue')
    plt.xlabel('Year')
    plt.ylabel('Number of Records')
    plt.title('Number of Records per Year')
    plt.show()


st.markdown(    """Data has wrong years so we should analyse data from 1856-1865""")


with st.echo():
    #we should analyse data from 1856-1865
    yearly_counts = dict(df['Year'].value_counts().sort_index())
    yearly_counts = pd.Series({i: yearly_counts[i] for i in yearly_counts if int(i) >= 1856 and int(i) <= 1865})
    
    # Create a bar graph
    plt.bar(yearly_counts.index, yearly_counts.values, color='skyblue')
    plt.xlabel('Year')
    plt.ylabel('Number of Records')
    plt.title('Number of Records per Year')
    plt.show()


st.markdown(    """##Price analysis""")


with st.echo():
    plt.figure(figsize=(8, 6))
    plt.hist(df[df['Price'] < 5000]['Price'], bins=10, color='skyblue') # To avoid data outliers
    plt.xlabel('Price')
    plt.ylabel('Frequency')
    plt.title('Price Distribution')
    plt.show()


with st.echo():
    df['Price'].describe()


with st.echo():
    df['Price'].median()


st.markdown(    """So mean price is 1311, standart deviation is 2160, median is 1200""")


st.markdown(    """##Age distribution""")


with st.echo():
    plt.figure(figsize=(8, 6))
    plt.hist(df['Age'], bins=10, edgecolor='k', color='skyblue') # To avoid data outliers
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.title('Age Distribution')
    plt.show()


st.markdown(    """From this bar plot its obvious that age 20 is the most popular""")


with st.echo():
    df['Age'].describe()


with st.echo():
    df['Age'].median()


st.markdown(    """So the mean age is 26, standart deviation is 11, median price is 23""")


st.markdown(    """#Sex and color distribution""")


with st.echo():
    plt.figure(figsize=(5, 5))
    plt.hist(df[df['Sex'].isin(['F', 'M'])]['Sex'], bins=2, edgecolor='k', color='skyblue') # To avoid data outliers
    plt.xlabel('Sex')
    plt.ylabel('Frequency')
    plt.title('Sex Distribution')
    plt.show()


with st.echo():
    color_counts = df['Color'].value_counts()
    
    total_count = len(df)
    percentage_threshold = 1
    color_counts = color_counts[color_counts / total_count * 100 >= percentage_threshold]
    
    plt.figure(figsize=(8, 8))
    plt.pie(color_counts, labels=color_counts.index, autopct='%1.1f%%', startangle=140)
    plt.title('Color Distribution')
    plt.axis('equal')
    plt.show()


st.markdown(    """#Data transformation""")


with st.echo():
    df['Sex_Code'] = df['Sex'].map({'F': 0, 'M': 1})
    df = df[df['Sex'].isin(['F', 'M'])]
    
    df['Color_Code'] = df['Color'].map({'Negro': 0, 'Mulatto': 1,'Griff': 2, 'Black': 3})
    df = df[df['Color'].isin(['Negro', 'Mulatto', 'Griff', 'Black'])]
    df


st.markdown(    """#Detailed overview
Here Ill calculate annual statistics and statistics for each color of skin""")


with st.echo():
    #annual statistics
    
    cut_df = df[(df['Year'] >= 1856) & (df['Year'] <= 1861)]
    
    
    annual_stat = {}
    for year in range(1856, 1862):
        annual_stat[year] = cut_df[cut_df['Year'] == year].describe()


with st.echo():
    plt.bar(range(1856, 1862), [annual_stat[i]['Price']['mean'] for i in annual_stat], color='skyblue')
    plt.xlabel('Year')
    plt.ylabel('Price')
    plt.title('Annual mean price')
    plt.show()


st.markdown(    """We can see gradual growth of mean price from 1856 until 1860 and then a drop""")


with st.echo():
    plt.bar(range(1856, 1862), [annual_stat[i]['Age']['mean'] for i in annual_stat], color='skyblue')
    plt.xlabel('Year')
    plt.ylabel('Age')
    plt.title('Annual mean age')
    plt.show()


st.markdown(    """Mean age of slaves doesnt change, just a little fluctuations""")


with st.echo():
    #skin color statistics
    color_stat = {}
    
    for color in ['Negro', 'Mulatto', 'Griff', 'Black']:
        color_group = df[df['Color'] == color]
        if not color_group.empty:
            color_stat[color] = color_group.describe()
    
    color_stat = pd.DataFrame({color: [color_stat[color]['Price']['mean'], color_stat[color]['Age']['mean']] for color in ['Negro', 'Mulatto', 'Griff', 'Black']})
    color_stat = color_stat.rename(index={0: 'Price', 1: 'Age'})


with st.echo():
    color_stat


st.markdown(    """The most valueable color is Black, the mean age is 24-26 for each skin color""")


st.markdown(    """#Hypothesis checking
My hypothesis is "Is correlation and distribution between age and price the same for Male and Female?"""")


with st.echo():
    male_df = df[df['Sex'] == 'M']
    female_df = df[df['Sex'] == 'F']


with st.echo():
    male_df


with st.echo():
    female_df


st.markdown(    """Firstly lets show mean price for each age group""")


with st.echo():
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


with st.echo():
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


st.markdown(    """We can see that distributions of males and females are similar. The most valuable group for females is 15-20 and for males is 20-25

Now lets calculate the correlation""")


with st.echo():
    correlation_female = female_df['Age'].corr(female_df['Price'])
    correlation_male = male_df['Age'].corr(male_df['Price'])
    
    print(f'Correlation between Age and Price for Female slaves: {correlation_female}')
    print(f'Correlation between Age and Price for Male slaves: {correlation_male}')


st.markdown(    """As we can see, there is no correlation for male and female, but lets calculate it for 0-20 and 20-80 age periods""")


with st.echo():
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


st.markdown(    """Here we can see correlation between age and price for 0-20 age groups, but they are different for males and females. For females age is more important.

Distributions are the same, but correlations are not, so hypothesis is false.""")

if __name__ == "__main__":
    main()
