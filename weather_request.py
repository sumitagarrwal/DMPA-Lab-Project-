import requests
import json
import datetime
import csv
import pandas as pd


API_KEY = 'f6c3b25b3687879ad85524eaa5d5bd7a'  #TODO make enviorment variable
BASE_URL = 'https://api.darksky.net/forecast/{}/{},{},{}?exclude=currently,minutely,hourly,alerts,flags&units=si'
# https://api.darksky.net/forecast/[key]/[latitude],[longitude],[time]


def get_weather(latitude, longitude, time):
    url = BASE_URL.format(API_KEY, latitude, longitude, time)
    response = requests.get(url)
    data = response.json()['daily']['data'][0]
    # print(json.dumps(data, indent=4))
    # df = pd.DataFrame.from_dict(data)
    # print(df)
    # print(data)
    return data


def write_to_csv(data):
    with open('manipal_weather.csv', 'a') as f:
        w = csv.DictWriter(f, data.keys())
        w.writerow(data)


def write_csv_header(data):
    with open('manipal_weather.csv', 'a') as f:
        w = csv.DictWriter(f, data.keys())
        w.writeheader()


def main():
    latitude = 13.3526  # MIT co-ordinates
    longitude = 74.7928
    start_datetime = int(datetime.datetime(2018, 1, 1).timestamp())
    end_datetime = int(datetime.datetime(2019, 1, 1).timestamp())
    for time in range(start_datetime, end_datetime, 86400):
        try:
            data = get_weather(latitude, longitude, time)
            if time == start_datetime:
                write_csv_header(data)
            write_to_csv(data)
            print(int((time-start_datetime)/86400)+1, "\t", time)
        except KeyError:
            print("Data not available for time: ", time)
        except:
            print("Error occurred at time: ", time)


if __name__ == '__main__':
    main()
