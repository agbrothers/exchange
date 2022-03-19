
from create_db import DataBase



if __name__=="__main__":

    database = DB("database/credentials.json")


    table = "AAPL"
    query = f"""
    select * from {table}
    where date = '2020-11-02';
    """

    dt = database.read(query)




