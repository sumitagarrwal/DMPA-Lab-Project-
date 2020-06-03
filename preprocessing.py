import sklearn
import pandas as pd


def read_data(filename):
    df = pd.read_csv(filename)
    return df


def filter_attributes(df):
    newdf = df[['temperatureHigh', 'temperatureLow', 'humidity', 'precipIntensityMax', 'precipProbability', 'windSpeed', 'cloudCover']]
    #print(newdf.head())
    return newdf


def outlier_removal(df):
    pass


def split_dataset(df):
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test
    # TODO split randomly
    #header = df.columns.values.tolist()
    df_train = df[:250]
    df_test = df[250:]
    print(df_train.head())
    print(df_test.head())
    df_train.to_csv('training.csv', index=False)
    df_test.to_csv('testing.csv', index=False)


def main():
    data = read_data('manipal_weather.csv')
    #print(data.head())
    data = filter_attributes(data)
    split_dataset(data)


if __name__ == '__main__':
    main()
