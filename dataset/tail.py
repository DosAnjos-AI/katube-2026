import pandas as pd

def dataset():
    df = pd.read_csv('dataset.csv',sep='|')
    print(df.tail())


def main():
    dataset()


if __name__ == "__main__":
    main()