from telegram import Update, Bot
from telegram.ext import Updater, CommandHandler, Filters, CallbackContext
import matplotlib.pyplot as plt
import pandas as pd
import io
import os

# Enable logging
import logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
df = pd.read_csv('CalomirisPritchett_data.csv')

df = df[['Sales Date', 'Sex', 'Age', 'Color', 'Price']]
df['Date'] = pd.to_datetime(df['Sales Date'], format='%m/%d/%Y', errors='coerce')
df['Price'].replace('.', pd.NA, inplace=True)
df['Sex'].replace('.', pd.NA, inplace=True)
df['Color'].replace('.', pd.NA, inplace=True)
df['Sales Date'].replace('.', pd.NA, inplace=True)

df = df.dropna(subset=['Price', 'Sex', 'Age', 'Date'])
df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
df['Age'] = pd.to_numeric(df['Age'], errors='coerce')

df['Year'] = df['Date'].dt.year

def start(update: Update, context: CallbackContext) -> None:
    update.message.reply_text('Welcome to the New Orleans Slave Sales Analysis Bot\n'
                              'Type /get_annual for annual statistics\n'
                              'Type /get_color for skin color distribution\n'
                              'Type /get_age for records by age\n'
                              'Type /get_period for mean price in a selected period')

def get_annual(update: Update, context: CallbackContext) -> None:
    cut_df = df[(df['Year'] >= 1856) & (df['Year'] <= 1861)]
    annual_stat = {}
    for year in range(1856, 1862):
        annual_stat[year] = cut_df[cut_df['Year'] == year].describe()

    fig, ax = plt.subplots()
    ax.bar(range(1856, 1862), [annual_stat[i]['Price']['mean'] for i in annual_stat], color='skyblue')
    ax.set_xlabel('Year')
    ax.set_ylabel('Price')
    ax.set_title('Annual mean price')


    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    context.bot.send_photo(chat_id=update.effective_chat.id, photo=buffer)

def get_color(update: Update, context: CallbackContext) -> None:
    color_counts = df['Color'].value_counts()
    total_count = len(df)
    percentage_threshold = 1
    color_counts = color_counts[color_counts / total_count * 100 >= percentage_threshold]

    fig, ax = plt.subplots()
    ax.pie(color_counts, labels=color_counts.index, autopct='%1.1f%%', startangle=140)
    ax.set_title('Color Distribution')
    ax.axis('equal')

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    context.bot.send_photo(chat_id=update.effective_chat.id, photo=buffer)

def get_age(update: Update, context: CallbackContext) -> None:
    try:
        age = int(context.args[0])  # Retrieve the age from the command argument
        age_count = df[df['Age'] == age].shape[0]
        update.message.reply_text(f'Number of records for age {age}: {age_count}')
    except (IndexError, ValueError):
        update.message.reply_text('Usage: /get_age <age>')

def get_period(update: Update, context: CallbackContext) -> None:
    try:
        start_year, end_year = int(context.args[0]), int(context.args[1])
        period_df = df[(df['Year'] >= start_year) & (df['Year'] <= end_year)]
        mean_price = period_df['Price'].mean()
        update.message.reply_text(f'Mean price from {start_year} to {end_year}: {mean_price}')
    except (IndexError, ValueError):
        update.message.reply_text('Usage: /get_period <start_year> <end_year>')

def error(update: Update, context: CallbackContext) -> None:
    logger.warning('Update "%s" caused error "%s"', update, context.error)

def main():

    api_token = '6954241837:AAF4px-xWLQ-Wp0L9Vs_defzE9R0pr-MyEE'
    updater = Updater(api_token, use_context=True)


    dp = updater.dispatcher


    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("get_annual", get_annual))
    dp.add_handler(CommandHandler("get_color", get_color))
    dp.add_handler(CommandHandler("get_age", get_age, pass_args=True))
    dp.add_handler(CommandHandler("get_period", get_period, pass_args=True))

 
    dp.add_error_handler(error)


    updater.start_polling()


    updater.idle()

if __name__ == '__main__':
    main()
