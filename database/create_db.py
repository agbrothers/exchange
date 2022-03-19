import os
import glob
import json
import pymysql
import numpy as np
import pandas as pd
from tqdm import trange


class DataBase:

    def __init__(self, credentials_path="database/credentials.json" ):
        # Opening JSON file
        f = open(credentials_path)
        credentials = json.load(f)
        self.host=credentials["host"]
        self.user=credentials["user"]
        self.password=credentials["password"]
        self.database=credentials["database"]

    def connect(self):
        conn = pymysql.connect(
            host=self.host,
            user=self.user,
            password=self.password,
            database=self.database,
        )
        return conn

    """""""""""""""""""""""""""
    CRUD FUNCTIONS
    """""""""""""""""""""""""""
    def create_database(self):
        conn = pymysql.connect(
            host=self.host,
            user=self.user,
            password=self.password,
        )
        conn.cursor().execute(f'create database {self.database}')
        conn.close()

    def query(self, query):
        try:
            conn = self.connect()
            conn.cursor().execute(query)
            conn.commit()
            conn.close()
        except Exception as e:
            print(f'Update Error: {e}')

    def read(self, query):
        try:
            conn = self.connect()
            df = pd.read_sql(query, conn)
            conn.close()
            return df
        except Exception as e:
            print(f'Query Failed: \n{e}')

    def update(self, query):
        try:
            conn = self.connect()
            cursor = conn.cursor()
            cursor.execute(query)
            conn.close()
        except Exception as e:
            print(f'Update Error: {e}')
            
    def create_table(self, name):
        table_query = f"""
            create table if not exists {name}(
                id int auto_increment not null,
                ticker varchar(5),

                timestamp datetime not null,
                date date not null,
                open float not null,
                high float not null,
                low float not null,
                close float not null,
                volume int not null,
                time time not null,

                primary key (id),
                foreign key (ticker) references tickers(ticker) on delete cascade
            );
            """        
        self.query(table_query)

    def drop_table(self, name):
        drop_query = f"DROP TABLE IF EXISTS {name};"
        self.query(drop_query)

    def insert(self, table, cols, values):
        values = self.none_to_null(values)
        query = self.add_cols(f'insert into {table}(', cols) + ') ' + self.add_cols('values(', values, as_str=True) + ');'
        self.query(query)

    def insert_multiple(self, table, cols, values):
        query = self.add_cols(f'insert into {table}(', cols) + ') values'
        for row in values:
            query += f" {tuple(row)},"
        query = query[:-1] + ';'
        self.query(query)

    def upload_csv(self, path):
        data = pd.read_csv(path)
        file = os.path.basename(path)
        table = file.split("_")[0]
        self.create_table(table)
        cols = data.keys().values

        batch_size = 500_000 # Uploading more than 500K at a time breaks the pipeline
        num_batches = np.ceil(data.shape[0] / batch_size) 
        count=0
        for batch in trange(int(num_batches)):
            if data.shape[0] - count < batch_size: 
                batch_size = data.shape[0] - count
            i = count
            j = count + batch_size
            values = data.iloc[i:j].values
            self.insert_multiple(table, cols, values)
            count += batch_size

    def autofill(self, cols, values, table):
        query = self.build_where(f'select id from {table} where ', cols, values) +';'
        result = self.read(query)
        if len(result) == 0: 
            return self.insert(cols, values, table)
        return result.iloc[0][0]

    def show_tables(self):
        tables = self.read('show tables;')
        for table in tables['Tables_in_solve']:
            print(f'\n{table}')
            print(self.read(f'select * from {table};').head())


    """""""""""""""""""""""""""
    QUERY BUILDING UTILS
    """""""""""""""""""""""""""
    def add_cols(self, query, cols, as_str=False):  
        # add a list of cols/values to a query string
        for i,item in enumerate(cols):
            query += str(item) if as_str==False or type(item)!=str or item=='null' else f"'{item}'"
            query += ', ' if i < len(cols)-1 else ''
        return query

    def build_where(self, query, cols, values):
        for i,col in enumerate(cols):
            query += str(col) +'='+str(values[i]) if type(values[i]) != str or values[i]=='null' else str(col) +"='"+str(values[i])+"'"
            query +=' and ' if i < len(cols)-1 else ''
        return query

    def build_row(self, cols, values):
        row = []
        for key in cols:
            if key in values.keys(): 
                row.append(values[key])
            else:
                row.append('null')                
        return row

    def none_to_null(self, values):
        nulls = pd.isnull(values)
        for i,item in enumerate(nulls):
            values[i] = 'null' if item == True else values[i]
        return values

    def check_props(self, cols, row):
        try:
            return [row[col] for col in cols]
        except Exception as e:
            return None
            




if __name__=="__main__":

    database = DB(
        user="root",
        password="sonny_crocket_84",
        database="exchange",
    )


    tickers_query = f"""
        create table if not exists tickers(
            ticker_id int auto_increment not null,
            ticker varchar(5) not null,
            first_date date,
            last_date date,

            primary key (ticker_id),
            unique (ticker)
        );
        """

    paths = glob.glob("./data/source/*.csv")
    for i,path in enumerate(paths):
        print(f"{i}/{len(paths)} {path}")
        database.upload_csv(path)

    df = database.read("SELECT table_name FROM information_schema.tables;")


    for path in paths:
        file = os.path.basename(path)
        table = file.split("_")[0]

        cols = ("ticker", "first_date", "last_date")
        mn = database.read(f"select min(date) from {table}").iloc[0]
        mn = str(mn[0])
        mx = database.read(f"select max(date) from {table}").iloc[0]
        mx = str(mx[0])
        values = [table, mn, mx]
        database.insert("tickers", cols, values)        

        # database.update("tickers", cols, values)        


    database.read("SELECT * FROM tickers;")



# database.read("SELECT * FROM tickers")

# path = "/Users/greysonbrothers/code/exchange/data/source/AAPL_1min.csv"

# database.drop_table("AAPL")

# database.upload_csv(path)

# aapl = database.read("SELECT * FROM AAPL")






