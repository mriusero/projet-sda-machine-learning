from dashboard import app_layout, load_data, merge_data

def main():

    data = load_data()
    dataframes = merge_data(data)

    app_layout(dataframes)

if __name__ == '__main__':
    main()
