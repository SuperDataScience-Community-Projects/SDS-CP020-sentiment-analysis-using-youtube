
# #dag - directed acyclic graph

# #tasks : 1) fetch amazon data (extract) 2) clean data (transform) 3) create and store data in table on postgres (load)
# #operators : Python Operator and PostgresOperator
# #hooks - allows connection to postgres
# #dependencies
from airflow import DAG
from airflow.providers.http.hooks.http import HttpHook
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.decorators import task
from airflow.utils.dates import days_ago
import requests
import json
import random
# Latitude and longitude for the desired location (London in this case)
LATITUDE = '51.5074'
LONGITUDE = '-0.1278'
# POSTGRES_CONN_ID='postgres_default'
# API_CONN_ID='open_meteo_api'

default_args={
    'owner':'airflow',
    'start_date':days_ago(1)
}

## DAG
with DAG(dag_id='weather_etl_pipeline',
         default_args=default_args,
         schedule_interval='@daily',
         catchup=False) as dags:
    
    @task()
    def extract_weather_data():
        """Extract weather data from Open-Meteo API using Airflow Connection."""

        # # Use HTTP Hook to get connection details from Airflow connection

        # http_hook=HttpHook(http_conn_id=API_CONN_ID,method='GET')

        # ## Build the API endpoint
        # ## https://api.open-meteo.com/v1/forecast?latitude=51.5074&longitude=-0.1278&current_weather=true
        # endpoint=f'/v1/forecast?latitude={LATITUDE}&longitude={LONGITUDE}&current_weather=true'

        # ## Make the request via the HTTP Hook
        # response=http_hook.run(endpoint)

        # if response.status_code == 200:
        #     return response.json()
        # else:
        #     raise Exception(f"Failed to fetch weather data: {response.status_code}")
        # """Generate sample weather data mimicking API response."""
        sample_weather_data = {
            "current_weather": {
                "temperature": round(random.uniform(-10, 40), 2),
                "windspeed": round(random.uniform(0, 50), 2),
                "winddirection": random.randint(0, 360),
                "weathercode": random.choice([0, 1, 2, 3])  # Example weather codes
            }
        }
        return sample_weather_data       
        
    @task()
    def transform_weather_data(weather_data):
        """Transform the extracted weather data."""
        current_weather = weather_data['current_weather']
        transformed_data = {
            'latitude': LATITUDE,
            'longitude': LONGITUDE,
            'temperature': current_weather['temperature'],
            'windspeed': current_weather['windspeed'],
            'winddirection': current_weather['winddirection'],
            'weathercode': current_weather['weathercode']
        }
        return transformed_data
    
    @task()
    def load_weather_data(transformed_data):
        """Load transformed data into PostgreSQL."""
        pg_hook = PostgresHook(postgres_conn_id='books_connection')
        conn = pg_hook.get_conn()
        cursor = conn.cursor()
        cursor.execute("""
                        SELECT * FROM pg_tables WHERE tablename = 'weather_data'; 
                        DROP TABLE IF EXISTS weather_data CASCADE;
                       """)
        # Create table if it doesn't exist
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS weather_data (
            latitude FLOAT,
            longitude FLOAT,
            temperature FLOAT,
            windspeed FLOAT,
            winddirection FLOAT,
            weathercode INT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)

        # Insert transformed data into the table
        cursor.execute("""
        INSERT INTO weather_data (latitude, longitude, temperature, windspeed, winddirection, weathercode)
        VALUES (%s, %s, %s, %s, %s, %s)
        """, (
            transformed_data['latitude'],
            transformed_data['longitude'],
            transformed_data['temperature'],
            transformed_data['windspeed'],
            transformed_data['winddirection'],
            transformed_data['weathercode']
        ))

        conn.commit()
        cursor.close()

    ## DAG Worflow- ETL Pipeline
    weather_data= extract_weather_data()
    transformed_data=transform_weather_data(weather_data)
    load_weather_data(transformed_data)





