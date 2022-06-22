import pandas as pd

def main():

    locales = [
        "us",
        "es",
        "jp",
    ]
    ##将三份不同语言的文件合并为同一份
    df = pd.DataFrame()
    for locale in locales:
        df_ = pd.read_csv(f"D:/KDD相关/任务1/task_1_ranking_model_{locale}.csv")
        df = pd.concat([df, df_])

    df.to_csv(f"D:/KDD相关/任务1/task_1_ranking_model.csv", index=False, sep=',',)


if __name__ == "__main__":
    main()

