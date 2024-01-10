import pandas as pd
import openmeteo_requests
import requests_cache
from retry_requests import retry
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
import os
import json
from dotenv import load_dotenv

#Load the configuration file containing the open-meteo api url and the location of the latitude and longitude. Also load the .env file.
def load_config():
    load_dotenv()

    with open('config.json') as json_file:
        data = json.load(json_file)
    return data

#Load the template for the llm prompt. This can be edited by the user to fit its need.
def load_template():
    with open('template.txt', 'r') as file:
        data = file.read().replace('\n', '')
    return data

#Setup the open-meteo client and the openAI llm.
def setup_api_clients(url):
    # Setup the Open-Meteo API client with cache and retry on error
    cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
    retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
    openmeteo = openmeteo_requests.Client(session = retry_session)

    # Setup the openAI API
    OPENAI_API_KEY = os.getenv('OPENAI_KEY')
    llm = OpenAI(openai_api_key = OPENAI_API_KEY)

    return openmeteo, url, llm

#Call the open-meteo api.
def get_weather_data(openmeteo, url, latitude, longitude):
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": ["temperature_2m", "precipitation_probability", "precipitation"],
    }

    responses = openmeteo.weather_api(url, params=params)

    return responses[0]


#Process the output of the open-meteo api and select the day's range
def process_hourly_data(response):
    # Process hourly data.
    hourly = response.Hourly()
    hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
    hourly_precipitation_probability = hourly.Variables(1).ValuesAsNumpy()
    hourly_precipitation = hourly.Variables(2).ValuesAsNumpy()

    hourly_data = {"date": pd.date_range(
        start = pd.to_datetime(hourly.Time(), unit = "s"),
        end = pd.to_datetime(hourly.TimeEnd(), unit = "s"),
        freq = pd.Timedelta(seconds = hourly.Interval()),
        inclusive = "left"
    )}
    hourly_data["temperature_2m"] = hourly_temperature_2m
    hourly_data["precipitation_probability"] = hourly_precipitation_probability
    hourly_data["precipitation"] = hourly_precipitation

    # Keep only the informations from 8AM to 8PM for the day
    hourly_dataframe = pd.DataFrame(data = hourly_data)
    hourly_dataframe['date'] = pd.to_datetime(hourly_dataframe['date'])
    hourly_dataframe = hourly_dataframe.set_index('date')
    hourly_dataframe = hourly_dataframe.between_time('08:00', '20:00')
    hourly_dataframe = hourly_dataframe.head(13)
    hourly_str = hourly_dataframe.to_string

    return hourly_str

#generate a one shot prompt containing the relevant weather data and call the openAI api
def generate_weather_summary(llm, hourly_str):
    template = load_template()
    weather_prompt_template = PromptTemplate.from_template(template)
    weather_prompt = weather_prompt_template.format(forecast=hourly_str)
    weather_summary = llm.generate([weather_prompt])

    return weather_summary.generations[0][0].text


def main():
    config = load_config()
    openmeteo, url, llm = setup_api_clients(config['api_url'])
    response = get_weather_data(openmeteo, url, config['latitude'], config['longitude'])
    hourly_str = process_hourly_data(response)
    weather_summary = generate_weather_summary(llm, hourly_str)
    print(weather_summary)

if __name__ == "__main__":
    main()
