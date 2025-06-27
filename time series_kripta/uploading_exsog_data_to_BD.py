import pandas as pd
from datetime import datetime, timedelta
import psycopg2
from psycopg2 import sql
import yfinance as yf

# Изменяемые параметры экзогенных данных и начала времени считывания
tickers = ['DX-Y.NYB', '^GSPC', 'BZ=F'] 
start_date = "2025-01-01" 

# Ключ - для создания full_date_range со всеми датами. Значение - для считывания данных с сайта yfinance в df_ticker
time_intervals = {'d': '1d', 
                  'h': '1h', 
                  '15min': '15m', 
                  'min': '1m'}

decoding_ticker = {    # Словарь расшифровки тикеров
    'DX-Y.NYB': 'DXY',
    '^GSPC': 'GSPC',
    'BZ=F': 'Oil_Brent'
}

POSTGRES_CONFIG = {     # параметры соединения с БД
    'dbname': 'postgres',
    'user': 'postgres',
    'password': '3253',
    'host': 'localhost',
    'port': '5432'
}


def upload_exsog_data_to_df(tickers: list, start_date: str, end_date: str, time_interval_key: str, time_interval_value: str) -> pd.DataFrame:
    """ Загрузка данных с помощью Yahoo Finance для указанного интервала времени.
        Parametrs:
            tickers (list): Список тикеров, экзогенных данных
            start_date (str): Начальная дата в формате "YYYY-MM-DD"
            end_date (str): Конечная дата в формате "YYYY-MM-DD"
            time_interval_key (str): Время между записями (время одной свечи) для создания дат в full_date_range
            time_interval_value (str): Время между записями (время одной свечи) для загрузки таргетов в df_ticker
        Returns:
            df_tickers (pd.DataFrame): Датафрейм с загруженными экзогенными данными друг под другом для определенного интервала
    """

    full_date_range = pd.date_range(start=start_date, end=end_date, freq=time_interval_key) # создаем список всех дат с выходными
    df_tickers = pd.DataFrame()

    for ticker in tickers:
        df_ticker = yf.download(ticker, start=start_date, end=end_date, interval=time_interval_value)
        df_ticker = df_ticker.reindex(full_date_range) # добавляем выходные заполненные пропусками 
        df_ticker.columns = df_ticker.columns.droplevel(1) # удаляем ненужный столбец
        df_ticker = df_ticker.reset_index() # делаем столбцом timestamp
        df_ticker = df_ticker.rename(columns={'index': 'timestamp'})
        df_ticker['code_dim_regressors'] = decoding_ticker[ticker] # создаем столбец с названием регрессора
        df_tickers = pd.concat([df_tickers, df_ticker], ignore_index=True)
    
    df_tickers = df_tickers.fillna(0) # заполняем пропуски по выходным 0 
    return df_tickers


def insert_data(conn: psycopg2.extensions.connection, time_intervals: str, df: pd.DataFrame) -> None:
    """
    Вставляет данные в таблицу PostgreSQL
    Parametrs:
        conn (psycopg2.extensions.connection): Соединение с PostgreSQL
        time_intervals (str): Интервал свечей. 1m, 15m, 1h, 1d
        df (pd.DataFrame): Таблица с данными
        code_dim_exog (str): Наименование валюты
        stock_market_code (str): Наименование биржи
    """
    # проверяем текущий интервал для образения к нужно таблице
    if time_intervals == '15m':
        time_intervals = '15_m'
    elif time_intervals == '1m':
        time_intervals = 'm'
    elif time_intervals == '1h':
        time_intervals = 'h'
    elif time_intervals == '1d':
        time_intervals = 'd'

    table_name = f"fct_regressors_{time_intervals}"

    data_to_insert = [] # список для хранения экзогенных данных в формате для SQL

    for _, row in df.iterrows(): # построчно заполняем список data_to_insert
        data_to_insert.append((
            datetime.now(), # время записи данных в БД
            str(row['code_dim_regressors']), 
            row['timestamp'],
            float(row['Open']), 
            float(row['High']),
            float(row['Low']),
            float(row['Close']),
            float(row['Volume'])
        ))

    # SQL запрос
    insert_sql = sql.SQL("""
        INSERT INTO exogenous_data.{} 
        (insert_date, code_dim_regressors, timestamp, open, high, low, close, volume)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """).format(sql.Identifier(table_name))

    # запись значений в БД
    with conn.cursor() as cursor:
        cursor.executemany(insert_sql, data_to_insert)
        conn.commit()
        

# общий вызов функций для записи данных в БД
end_date = datetime.now() # 2025-05-04 16:10:25.024458

# устанавливаем связь с БД
conn = psycopg2.connect(**POSTGRES_CONFIG)
print("Успешное подключение к PostgreSQL")

for time_interval_key, time_interval_value in time_intervals.items(): # с помощью переборов проверим все таблицы на заполненность до тек. даты
    if time_interval_value == '15m':
        time_intervals_for_name_table = '15_m'
    elif time_interval_value == '1m':
        time_intervals_for_name_table = 'm'
    elif time_interval_value == '1h':
        time_intervals_for_name_table = 'h'
    elif time_interval_value == '1d':
        time_intervals_for_name_table = 'd'
        
  
    table_name = f"fct_regressors_{time_intervals_for_name_table}"
    cursor = conn.cursor()
    last_record_query = sql.SQL("""
        SELECT MAX(timestamp) FROM exogenous_data.{}
        WHERE code_dim_regressors = %s 
    """).format(sql.Identifier(table_name))

    cursor.execute(last_record_query, ('Oil_Brent',))
    last_record = cursor.fetchone()
    last_record_date = last_record[0]
    print(last_record_date)
    if last_record_date == None: # None означает, что данных в БД нет, поэтому нужно их загрузить с нуля
        print('В БД нет данных, но сейчас мы это исправим')
        end_date = end_date - timedelta(days=1)
        # проверяем текущий интервал, для которого нужно будет заполнить таблицу
        if time_interval_key == 'd':
            exog_data = upload_exsog_data_to_df(tickers, start_date, end_date, time_interval_key, time_interval_value)
            print(time_interval_value)
            print(exog_data)
            
        elif time_interval_key == 'h': 
            start_date = end_date - timedelta(days=720) # начинаем считывать данные за 720 дней из-за ограничения на yfinance
            exog_data = upload_exsog_data_to_df(tickers, start_date, end_date, time_interval_key, time_interval_value)
            print(time_interval_value)
            print(exog_data)
            
        elif time_interval_key == '15min':
            start_date = end_date - timedelta(days=60)
            exog_data = upload_exsog_data_to_df(tickers, start_date, end_date, time_interval_key, time_interval_value)
            print(time_interval_value)
            print(exog_data)
            
        elif time_interval_key == 'min':
            start_date = end_date - timedelta(days=7)
            exog_data = upload_exsog_data_to_df(tickers, start_date, end_date, time_interval_key, time_interval_value)
            print(time_interval_value)
            print(exog_data)
        
        else:
            print('Проверьте правильность заполнения словаря time_intervals')
        
        # заполняем таблицы в БД
        insert_data(conn, time_interval_value, exog_data)
            
    elif end_date > last_record_date: # проверяем последнюю запись с текущим временем
                # проверяем текущий интервал, для которого нужно будет заполнить таблицу
        if time_interval_key == 'd':
            exog_data = upload_exsog_data_to_df(tickers, last_record_date, end_date, time_interval_key, time_interval_value)
            print(time_interval_value)
            print(exog_data)
            
        elif time_interval_key == 'h': 
            start_date = end_date - timedelta(days=720) # начинаем считывать данные за 720 дней из-за ограничения на yfinance
            exog_data = upload_exsog_data_to_df(tickers, last_record_date, end_date, time_interval_key, time_interval_value)
            print(time_interval_value)
            print(exog_data)
            
        elif time_interval_key == '15min':
            start_date = end_date - timedelta(days=60)
            exog_data = upload_exsog_data_to_df(tickers, last_record_date, end_date, time_interval_key, time_interval_value)
            print(time_interval_value)
            print(exog_data)
            
        elif time_interval_key == 'min':
            start_date = end_date - timedelta(days=7)
            exog_data = upload_exsog_data_to_df(tickers, last_record_date, end_date, time_interval_key, time_interval_value)
            print(time_interval_value)
            print(exog_data)
        
        # заполняем таблицы в БД
        insert_data(conn, time_interval_value, exog_data)
        

    
    
