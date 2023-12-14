#!/usr/bin/env python3
# coding: utf-8

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt


@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path)

df = load_data('CalomirisPritchett_data.csv')


def preprocess_data(df):
    df = df[['Sales Date', 'Sex', 'Age', 'Color', 'Price']]
    df = df.rename(columns=lambda x: x.strip())
    df['Date'] = pd.to_datetime(df['Sales Date'], format='%m/%d/%Y', errors='coerce')
    df = df.sort_values(by='Date')
    df['Price'].replace('.', pd.NA, inplace=True)
    df['Sex'].replace('.', pd.NA, inplace=True)
    df['Color'].replace('.', pd.NA, inplace=True)
    df['Sales Date'].replace('.', pd.NA, inplace=True)
    df.dropna(subset=['Price', 'Sex', 'Age', 'Date'], inplace=True)
    df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
    df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
    return df

df = preprocess_data(df)


def create_app(df):
    st.title("New Orlean's Slave Sales Analysis")


    st.markdown("### Data Description")
    st.text("Sales Date - The date of the transaction (MM/DD/YYYY format). This is NOT the date the sale was recorded in the conveyance office.")
    st.text("Sex Male (M) or female (F). - Gender often inferred from sale record.")
    st.text("Age - Age in years")
    st.text("Color - Description of the slave’s skin color, may also indicate ancestry.")
    st.text("Price - Price associated with slaves. For group sales with a single price, only one price is listed for the first slave.")
    st.write(df.head(15))

    st.markdown("### Data Overview")
    st.write(df.describe())


    df['Year'] = df['Date'].dt.year
    yearly_counts = df['Year'].value_counts().sort_index()

    yearly_counts = yearly_counts[(yearly_counts.index >= 1856) & (yearly_counts.index <= 1865)]

    fig_years = px.bar(yearly_counts, x=yearly_counts.index, y=yearly_counts.values, title="Number of Records per Year")
    fig_years.update_layout(xaxis_title="Year", yaxis_title="Number of Records")
    st.plotly_chart(fig_years)


    st.markdown("### Price Analysis")
    st.write(df['Price'].describe())
    st.write(f"Median price is: {df['Price'].median()}")


    fig_price = px.histogram(df[df['Price'] < 5000], x='Price', nbins=10, title="Price Distribution")
    fig_price.update_layout(xaxis_title="Price", yaxis_title="Frequency")
    st.plotly_chart(fig_price)

    st.markdown("Most slaves are sold for a price around the median of 1200.")


    st.markdown("### Age Distribution")
    st.write(df['Age'].describe())
    st.write(f"Median age is: {df['Age'].median()}")


    fig_age = px.histogram(df, x='Age', nbins=10, title="Age Distribution")
    fig_age.update_layout(xaxis_title="Age", yaxis_title="Frequency")
    st.plotly_chart(fig_age)

    st.markdown("Age 20 is the most common. The mean age is 26, standard deviation is 11.")


    cut_df = df[(df['Year'] >= 1856) & (df['Year'] <= 1861)]
    annual_stat = {year: cut_df[cut_df['Year'] == year].describe() for year in range(1856, 1862)}


    fig_annual_price = px.bar(
        x=range(1856, 1862),
        y=[annual_stat[i]['Price']['mean'] for i in annual_stat],
        title="Annual Mean Price"
    )
    fig_annual_price.update_layout(xaxis_title="Year", yaxis_title="Price")
    st.plotly_chart(fig_annual_price)


    fig_annual_age = px.bar(
        x=range(1856, 1862),
        y=[annual_stat[i]['Age']['mean'] for i in annual_stat],
        title="Annual Mean Age"
    )
    fig_annual_age.update_layout(xaxis_title="Year", yaxis_title="Age")
    st.plotly_chart(fig_annual_age)
    
    st.markdown("### Skin Color Distribution")
    color_counts = df['Color'].value_counts()
    total_count = len(df)
    percentage_threshold = 1
    color_counts = color_counts[color_counts / total_count * 100 >= percentage_threshold]


    fig_color = px.pie(color_counts, names=color_counts.index, values=color_counts, title='Color Distribution')
    fig_color.update_traces(textinfo='percent+label')
    st.plotly_chart(fig_color)
    

    color_stat = {color: df[df['Color'] == color].describe() for color in df['Color'].unique() if not df[df['Color'] == color].empty}
    
    color_stat_df = pd.DataFrame({
        color: [color_stat[color]['Price']['mean'], color_stat[color]['Age']['mean']]
        for color in color_stat
    }).rename(index={0: 'Price', 1: 'Age'})
    
    st.dataframe(color_stat_df)
    
    st.markdown("Black is the most valuable color category, with mean ages across skin colors ranging from 24-26.")

    st.markdown("### Hypothesis Checking")
    st.markdown('Hypothesis: "Is correlation and distribution between age and price the same for Male and Female?"')

    male_df = df[df['Sex'] == 'M']
    female_df = df[df['Sex'] == 'F']

    age_bins = pd.interval_range(start=0, end=int(max(male_df['Age'].max(), female_df['Age'].max())), freq=5)
    male_df['Age_Group'] = pd.cut(male_df['Age'], bins=age_bins).astype(str)
    female_df['Age_Group'] = pd.cut(female_df['Age'], bins=age_bins).astype(str)

    male_corr = male_df['Age'].corr(male_df['Price'])
    female_corr = female_df['Age'].corr(female_df['Price'])

    st.write(f'Correlation between Age and Price for Female slaves: {female_corr:.3f}')
    st.write(f'Correlation between Age and Price for Male slaves: {male_corr:.3f}')

    
    age_group_mean_price_female = female_df[female_df['Price'] < 1500].groupby('Age_Group')['Price'].mean()
    plt.figure(figsize=(10, 6))
    age_group_mean_price_female.plot(kind='bar', color='skyblue')
    plt.title('Mean Price of Female Slaves in Each Age Group')
    plt.xlabel('Age Group')
    plt.ylabel('Mean Price')
    plt.xticks(rotation=45)
    
    st.pyplot(plt)

    age_group_mean_price_male = male_df[male_df['Price'] < 5000].groupby('Age_Group')['Price'].mean()
    plt.figure(figsize=(10, 6))
    age_group_mean_price_male.plot(kind='bar', color='skyblue')
    plt.title('Mean Price of Male Slaves in Each Age Group')
    plt.xlabel('Age Group')
    plt.ylabel('Mean Price')
    plt.xticks(rotation=45)
    
    st.pyplot(plt)

    male_df_age_0_20 = male_df[(male_df['Age'] >= 0) & (male_df['Age'] <= 20)]
    male_df_age_20_80 = male_df[(male_df['Age'] > 20) & (male_df['Age'] <= 80)]

    female_df_age_0_20 = female_df[(female_df['Age'] >= 0) & (female_df['Age'] <= 20)]
    female_df_age_20_80 = female_df[(female_df['Age'] > 20) & (female_df['Age'] <= 80)]

    male_corr_0_20 = male_df_age_0_20['Age'].corr(male_df_age_0_20['Price'])
    male_corr_20_80 = male_df_age_20_80['Age'].corr(male_df_age_20_80['Price'])

    female_corr_0_20 = female_df_age_0_20['Age'].corr(female_df_age_0_20['Price'])
    female_corr_20_80 = female_df_age_20_80['Age'].corr(female_df_age_20_80['Price'])

    st.write(f'Correlation between Age (0-20) and Price for Male slaves: {male_corr_0_20:.3f}')
    st.write(f'Correlation between Age (20-80) and Price for Male slaves: {male_corr_20_80:.3f}')
    st.write(f'Correlation between Age (0-20) and Price for Female slaves: {female_corr_0_20:.3f}')
    st.write(f'Correlation between Age (20-80) and Price for Female slaves: {female_corr_20_80:.3f}')

    st.markdown("Distributions are the same, but correlations between age and price differ for males and females, therefore the hypothesis is not confirmed.")

if __name__ == "__main__":
    create_app(df)
