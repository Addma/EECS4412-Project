import enum
from xml.etree.ElementInclude import include
import pandas as pd
import csv
import matplotlib.pyplot as plt
import numpy as np
import json
from datetime import datetime
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from collections import Counter, defaultdict
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import train_test_split  


def main():
    df = pd.read_json("./resources/test.json")
    df = preprocessData(df)
    df.to_json("./resources/processed.json")
    df1 = pd.read_json("./resources/processed.json")
    df1.to_csv("./resources/processed.csv")
    convertToMarketBasket(df1)
def convertToMarketBasket(data):
    file = open("./resources/216480444-.csv", "w")
    writer = csv.writer(file)
    for index, row in data.iterrows():
        rowOut = []
        for ind, val in row.items():
            if val:
                rowOut.append(ind)
        writer.writerow(rowOut)
def confidence(data, item):
    i = 0
def processCategories(cat):
    res = cat.split(', ')
    res.remove('Restaurants')
    return res
   # print(len(data[data['stars'] <= 3]))
   # print(len(data[data['stars'] == 3.5]))
   # print(len(data[data['stars'] > 3.5]))
   # print(len(data['review_count']))
   # print(len(data[data['review_count']< 20]))
   # print(len(data[data['review_count'].between(20, 53, inclusive='both')]))
   # print(len(data[data['review_count'] > 53]))
def parseHours(h):
    if type(h) is float:
        return 0
    total = 0
    all_days = h.split(',')
    for day in all_days:
        times = day.split("-")
        times[0] = times[0].split('\'')[3]
        times[1] = times[1].replace('\'', '')
        times[1] = times[1].replace('}', '')
        time1 = datetime.strptime(times[0], "%H:%M")
        time2 = datetime.strptime(times[1], "%H:%M")
        total += (time2 - time1).seconds/3600
    return total
def getDays(hours):
    if type(hours) is float:
        return {"Monday": False, "Tuesday": False, "Wednesday": False, "Thursday": False, "Friday": False, "Saturday": False, "Sunday": False}
    all_days = hours.split(',')
    for day in all_days:
        times = day.split("-")
        times[0] = times[0].split('\'')[3]
        times[1] = times[1].replace('\'', '')
        times[1] = times[1].replace('}', '')
        time1 = datetime.strptime(times[0], "%H:%M")
        time2 = datetime.strptime(times[1], "%H:%M")
    
def supportCategories(rule, col, len):
    total = 0
    for record in col:
        if record == rule:
            total += 1
    return total / len
def checkTF(x):
    if x:
        return 1
    else:
        return 0
def binarizeAttrsCats(data, x):
    if data is None:
        return 0
    if x in data:
        return 1
    else:
        return 0
def preprocessData(data):
    df2 = pd.read_csv("./resources/temp.csv")
    data.index = df2.index
    data['total_hours'] = df2['total_hours']
    data['attributes'].dropna()
    data['categories'] = data['categories'].apply(lambda x: processCategories(x))
    arrayAllCategories = data['categories'].explode().unique()
    arrayAllAttributes = data['attributes'].explode().unique()
    top10Categories = dict.fromkeys(arrayAllCategories, 0)
    top10Attributes = dict.fromkeys(arrayAllAttributes, 0)
    i = 0
    for attribute in data['attributes']:
        if attribute == None:
            continue
        i += 1
        attribute = np.array(list(attribute.keys()))
        findAttr = np.isin(arrayAllAttributes, attribute)
        getAttr = np.nditer(findAttr, flags=['c_index'])
        while not getAttr.finished:
            if getAttr[0]:
                top10Attributes[arrayAllAttributes[getAttr.index]] += 1
            next = getAttr.iternext()
    for categories in data['categories']:
        categories = np.array(categories)
        findCategory = np.isin(arrayAllCategories, categories)
        getIndex = np.nditer(findCategory, flags=['c_index'])
        while not getIndex.finished:
            if getIndex[0]: 
                top10Categories[arrayAllCategories[getIndex.index]] += 1
            next = getIndex.iternext()
    top10Categories = dict(sorted(top10Categories.items(), key=lambda item: item[1], reverse=True)[:10])
    top10Attributes = dict(sorted(top10Attributes.items(), key=lambda item: item[1], reverse=True)[:10])

    with open('./resources/top10attributes.csv', 'w') as file:
        write = csv.writer(file)
        for key, value in top10Attributes.items():
            write.writerow([key, value])
    with open('./resources/top10categories.csv', 'w') as file:
        write = csv.writer(file)
        for key, value in top10Categories.items():
            write.writerow([key, value])
    for cat in top10Categories:
        data[cat] = data['categories'].apply(lambda x: binarizeAttrsCats(x, cat))
    for attr in top10Attributes:
        data[attr] = data['attributes'].apply(lambda x: binarizeAttrsCats(x, attr))
    data['stars <= 3'] = (data['stars'] <= 3).apply(lambda x : checkTF(x))
    data['stars = 3.5'] = (data['stars'] == 3.5).apply(lambda x: checkTF(x))
    data['stars > 3.5'] = (data['stars'] > 3.5).apply(lambda x: checkTF(x))
    data['review_count low'] = (data['review_count'] < 20).apply(lambda x: checkTF(x))
    data['review_count mid'] = data['review_count'].between(20, 53, inclusive='both').apply(lambda x: checkTF(x))
    data['review_count high'] = (data['review_count'] > 53).apply(lambda x: checkTF(x))
    data['total_hours low'] = (data['total_hours'] < 52).apply(lambda x: checkTF(x))
    data['total_hours mid'] = data['total_hours'].between(52, 74, inclusive='left').apply(lambda x: checkTF(x))
    data['total_hours high'] = (data['total_hours'] >= 74).apply(lambda x: checkTF(x))
    data = data.drop(columns=['business_id', 'name', 'address', 'city', 'stars', 'state', 'postal_code', 'latitude', 'longitude', 'attributes', 'categories', 'hours', 'review_count', 'total_hours'])
    return data
if __name__ == "__main__":
    main()