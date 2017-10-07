import matplotlib.style as style
import pandas as pd

if __name__ == '__main__':
    data_link = 'http://www.randalolson.com/wp-content/uploads/percent-bachelors-degrees-women-usa.csv'

    with open('percent-bachelors-degrees-women-usa.csv') as csvFile:
        majors = pd.read_csv(csvFile)

    if majors.size == 0:
        print('loaded from link\n')
        majors = pd.read_csv(data_link)
    else:
        print('loaded from file\n')

    # print(majors.info())
    # print("\n")
    # majors.head()
    # majors.tail()

    under_20 = majors.loc[0, majors.loc[0] < 20]
    # print(under_20)

    # under_20_graph = majors.plot(x = 'Year', y = under_20.index, figsize = (12, 8))
    # print('Type:', type(under_20_graph))
    style.use('fivethirtyeight')
    majors.plot(x='Year', y=under_20.index, figsize=(12, 8))
