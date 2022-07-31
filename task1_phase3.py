import enum
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
    df = pd.read_csv("./resources/test.csv")
    df3 = pd.read_json("./resources/yelp_academic_dataset_business.json", lines=True)
    df3 = df3[df3['business_id'].isin(df['business_id'])]
    print(df3['hours'])
    df3['hours'] = df['hours']
    df3.reset_index(inplace=True)
    findFreqItemSets(df3)
def support(data, item):
    i = 0
def confidence(data, item):
    i = 0
def processCategories(cat):
    res = cat.split(', ')
    res.remove('Restaurants')
    return res
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
def checkTakeOut(x):
    print(x['RestaurantsTakeOut'])
    if x is None:
        return None
    elif 'RestaurantsTakeOut' in x and x['RestaurantsTakeOut'] == True:
        return True
    else:
        return False
def findFreqItemSets(data):
    data['attributes'].dropna()
    print(data['attributes'])
    print(data['hours'])
    data['categories'] = data['categories'].apply(lambda x: processCategories(x))
    data['weekly_hours'] = data['hours'].apply(lambda x: parseHours(x))
    arrayAllCategories = data['categories'].explode().unique()
    arrayAllAttributes = data['attributes'].explode().unique()
    print(arrayAllAttributes)
    top20Categories = dict.fromkeys(arrayAllCategories, 0)
    top20Attributes = dict.fromkeys(arrayAllAttributes, 0)
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
                top20Attributes[arrayAllAttributes[getAttr.index]] += 1
            next = getAttr.iternext()
    for categories in data['categories']:
        categories = np.array(categories)
        findCategory = np.isin(arrayAllCategories, categories)
        getIndex = np.nditer(findCategory, flags=['c_index'])
        while not getIndex.finished:
            if getIndex[0]: 
                top20Categories[arrayAllCategories[getIndex.index]] += 1
            next = getIndex.iternext()
    top20Categories = dict(sorted(top20Categories.items(), key=lambda item: item[1], reverse=True)[:20])
    top20Attributes = dict(sorted(top20Attributes.items(), key=lambda item: item[1], reverse=True)[:20])
    data['stars > 4'] = data['stars'].apply(lambda x: x >= 4)
    data['3 < stars < 4'] = data['stars'].apply(lambda x: 3 <= x < 4)
    data['2 < stars < 3'] = data['stars'].apply(lambda x: 2 <= x < 3)
    data['1 < stars < 2'] = data['stars'].apply(lambda x: 1 <= x < 2)
    with open('./resources/top20attributes.csv', 'w') as file:
        write = csv.writer(file)
        for key, value in top20Attributes.items():
            write.writerow([key, value])
    with open('./resources/top20categories.csv', 'w') as file:
        write = csv.writer(file)
        for key, value in top20Categories.items():
            write.writerow([key, value])
    data['Takeout'] = data['attributes'].apply(lambda x: checkTakeOut(x))
    print(data['Takeout'] == True)
if __name__ == "__main__":
    main()