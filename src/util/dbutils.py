import psycopg2
import pandas as pd

from dotenv import load_dotenv
import os

env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
load_dotenv(dotenv_path=env_path)

DB_CONFIG = {
    'host': os.getenv('RDB_HOST'),
    'port': os.getenv('RDB_PORT'),
    'dbname': os.getenv('RDB_NAME'),
    'user': os.getenv('RDB_USER'),
    'password': os.getenv('RDB_PASSWORD')
}
def connect_db():
    """PostgreSQL DB 연결"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        return conn
    except Exception as e:
        print("❌ DB 연결 실패:", e)
        return None

def execute_query(conn, query):
    """쿼리 실행 후 DataFrame 반환"""
    try:
        df = pd.read_sql_query(query, conn)
        return df
    except Exception as e:
        print("❌ 쿼리 실행 실패:", e)
        return pd.DataFrame()

def close_db(conn):
    """DB 연결 종료"""
    if conn:
        conn.close()

def print_schema_description(df):
    schema_description = "\n".join([f"- {col}: {df[col].dtype}" for col in df.columns])
    print(schema_description)

def getCountryStaticSQLResult(query):
    conn = None
    try:
        conn = connect_db()
        if conn:
            result_df = execute_query(conn, query)
            print(result_df)
            print("\n컬럼별 데이터 타입:")
            print_schema_description(result_df)
            return result_df
    except Exception as e:
        print("❌ 오류 발생:", e)
        return "error:Rdb"
    finally:
        if conn:
            close_db(conn)

def getformationStaticSQLResult(query):
    conn = None
    try:
        conn = connect_db()
        if conn:
            result_df = execute_query(conn, query)
            print(result_df)
            print("\n컬럼별 데이터 타입:")
            print_schema_description(result_df)
            return result_df
    except Exception as e:
        print("❌ 오류 발생:", e)
        return "error:Rdb"
    finally:
        if conn:
            close_db(conn)

