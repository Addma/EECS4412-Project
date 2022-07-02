import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from collections import Counter
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import train_test_split
# Run any of the methods here. If method name has "user" in it, use the user_sample data.
def main():
    plt.rcParams['font.size'] = '14'
    plt.rcParams['figure.figsize'] = (12, 6)
    sample = pd.read_csv("./resources/test.csv")
    user_sample = pd.read_csv('./resources/user.csv')
    plotStarOccurrence(sample)
    plotReviewCountStars(sample)
    plotAttributeCount(sample)
    plotUser(user_sample)
    pca(sample)
    pcaUser(user_sample)
    
def plotStarOccurrence(sample):
    all_stars = sample['stars']
    ratings = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
    star_occurrences = {key: 0 for key in ratings}
    for stars in all_stars:
        star_occurrences[stars] += 1
    plt.plot(star_occurrences.keys(), star_occurrences.values(), '-ok')
    plt.xlabel('Star Rating')
    plt.ylabel('# of Occurrences')
    plt.title("Star Occurrences Distribution")
    plt.savefig('./resources/Restaurants_Star_Occurrence.png')
    plt.show()
def plotReviewCountStars(sample):
    all_reviewcount = sample['review_count']
    all_stars = sample['stars']
    c = Counter(zip(all_stars.values,all_reviewcount.values))
    s = [10*c[x1,y1] for x1,y1 in zip(all_stars.values,all_reviewcount.values)]
    plt.scatter(all_stars, all_reviewcount, s=s)
    plt.xlabel('Star rating')
    plt.ylabel('# of Reviews')
    plt.title("Review Count vs Star Rating")
    plt.savefig('./resources/Restaurants_Review_Count.png')
    plt.show()
def plotAttributeCount(sample):
    all_attributes = sample['attributes']
    all_stars = sample['stars']
    attributes_nums = {}
    i = 0
    for attr in all_attributes:
        if (isfloat(attr)):
            attributes_nums[i] = 0.1
        else:
            attributes_nums[i] = attr.count("True")
        i+=1
    c = Counter(zip(all_stars.values,attributes_nums.values()))
    s = [10*c[x1,y1] for x1,y1 in zip(all_stars.values,attributes_nums.values())]
    plt.scatter(all_stars, attributes_nums.values(), s=s)
    plt.xlabel("Star Rating")
    plt.ylabel("# of Attributes")
    plt.title("User Average Rating vs Number of Attributes")
    plt.savefig('./resources/Restaurants_Attributes_Stars.png')
    plt.show()
def plotUser(sample):
    avg_rating = sample['average_stars']
    ratings = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
    star_occurrences = {key: 0 for key in ratings}
    for stars in avg_rating:
        round_stars = round(stars * 2) / 2
        star_occurrences[round_stars] += 1
    all_fans = sample['fans']
    complimentCount = sample['compliment_more'] + sample['compliment_profile'] + \
    sample['compliment_cute'] + sample['compliment_list'] + sample['compliment_note'] + \
    sample['compliment_plain'] + sample['compliment_cool'] + sample['compliment_funny'] +\
    sample['compliment_writer'] + sample['compliment_photos'] + sample['compliment_hot']
    plt.scatter(avg_rating, all_fans)
    plt.title("Average Rating vs Fans")
    plt.xlabel('Average Rating')
    plt.ylabel('# of Fans')
    plt.savefig('./resources/user_rating_fans.png')
    plt.show()
    plt.plot(star_occurrences.keys(), star_occurrences.values(), '-ok') 
    plt.title("Star Rating Distribution")
    plt.savefig("./resources/user_rating_dist.png")
    plt.xlabel('Star Ratings')
    plt.ylabel('# of Occurrences')
    plt.show()
    plt.scatter(avg_rating, complimentCount)
    plt.title("Average Rating vs Compliment Count")
    plt.xlabel("Average Rating")
    plt.ylabel("# of Total Compliments")
    plt.savefig('./resources/user_rating_compliments.png')
    plt.show()
def isfloat(num):
    try:
        float(num)
        return True
    except ValueError:
        return False
def countAttrs(attrs):
        if isfloat(attrs):
            return 0
        else:
            return attrs.count("True")
# PCA 
def pca(sample):
    all_attributes = sample['attributes']
    all_hours = sample['hours']
    all_names = sample['name']
    attributes_nums = {}
    weekly_hours = {}
    i = 0
    for attr in all_attributes:
        attributes_nums[i] = countAttrs(attr)
        i+=1
    i=0
    for time in all_hours:
        weekly_hours[i] = parsehours(time)
        i+1
    pcaDF = sample[['stars', 'is_open', 'review_count']]
    pcaDF['weekly_hours'] = sample['hours'].apply(lambda x: parsehours(x))
    pcaDF['attribute_count'] = sample['attributes'].apply(lambda x: countAttrs(x))
    pcaDF.index = all_names
    pcaData = pcaDF
    scaled_data = StandardScaler().fit_transform(pcaDF.T)
    pca = PCA()
    pca.fit(scaled_data)
    pca_data = pca.transform(scaled_data)
    per_var = np.round(pca.explained_variance_ratio_* 100, decimals=1)
    labels = ['PC' + str(x) for x in range(1, len(per_var) + 1)]
    plt.bar(x=range(1, len(per_var)+1), height=per_var, tick_label=labels)
    plt.ylabel('Percentage of Explained Variance')
    plt.xlabel('Principle Component')
    plt.title("All PCA Variance Plot")
    plt.savefig('./resources/Restaurants_PCA_Figure_1.png')
    plt.show()
    pca_df = pd.DataFrame(pca_data, index=[pcaData[['stars', 'is_open', 'review_count', 'attribute_count', 'weekly_hours']]], columns=labels)
    plt.scatter(pca_df.PC1, pca_df.PC2)
    plt.xlabel('PC1 - {0}%'.format(per_var[0]))
    plt.ylabel('PC2 - {0}%'.format(per_var[1]))
    plt.title("PCA1&2 Plot")
    for sample in pca_df.index:
        plt.annotate(sample, (pca_df.PC1.loc[sample], pca_df.PC2.loc[sample]))
    plt.savefig('./resources/Restaurants_PCA_Figure_2.png')
    plt.show()

    loading_scores = pd.Series(pca.components_[0], index=all_names)
    loading_scores.columns = ['Name', 'Value']
    loading_scores2 = pd.Series(pca.components_[1], index=all_names)
    loading_scores2.columns = ['Name', 'Value']

    loading_scores.to_csv('./resources/restaurant-pca1.csv')
    loading_scores2.to_csv('./resources/restaurant-pcs2.csv')
    sorted_loading_scores = loading_scores.abs().sort_values(ascending=False)
    top_values = sorted_loading_scores[0:10].index.values
    print(loading_scores[top_values])
def calculateDate(dates):
    date = dates.split(' ')[0]
    today = datetime.strptime('2022-07-01', '%Y-%m-%d')
    accDate = datetime.strptime(date ,'%Y-%m-%d')
    return (today - accDate).days
def pcaUser(sample):
    all_names = sample['name']
    pcaDF = sample[['average_stars', 'fans']]
    pcaDF['compliment_count'] = sample['compliment_more'] + sample['compliment_profile'] + \
    sample['compliment_cute'] + sample['compliment_list'] + sample['compliment_note'] + \
    sample['compliment_plain'] + sample['compliment_cool'] + sample['compliment_funny'] +\
    sample['compliment_writer'] + sample['compliment_photos'] + sample['compliment_hot']
    pcaDF['votes'] = sample['useful'] + sample['funny'] + sample['cool']
    pcaDF['account_age'] = sample['yelping_since'].apply(lambda x: calculateDate(x))
    pcaDF.index = all_names
    pcaData = pcaDF
    scaled_data = StandardScaler().fit_transform(pcaData.T)
    pca = PCA()
    pca.fit(scaled_data)
    pca_data = pca.transform(scaled_data)
    per_var = np.round(pca.explained_variance_ratio_* 100, decimals=1)
    labels = ['PC' + str(x) for x in range(1, len(per_var) + 1)]
    plt.bar(x=range(1, len(per_var)+1), height=per_var, tick_label=labels)
    plt.ylabel('Percentage of Explained Variance')
    plt.xlabel('Principle Component')
    plt.title("Scree Plot")
    plt.savefig('./resources/User_PCA_Figure_1.png')
    plt.show()
    pca_df = pd.DataFrame(pca_data, index=[pcaData[['average_stars', 'fans', 'compliment_count', 'votes', 'account_age']]], columns=labels)
    plt.scatter(pca_df.PC1, pca_df.PC2)
    plt.xlabel('PC1 - {0}%'.format(per_var[0]))
    plt.ylabel('PC2 - {0}%'.format(per_var[1]))
    for sample in pca_df.index:
        plt.annotate(sample, (pca_df.PC1.loc[sample], pca_df.PC2.loc[sample]))
    plt.savefig('./resources/User_PCA_Figure_2.png')
    plt.show()

    loading_scores = pd.Series(pca.components_[0], index=all_names)
    loading_scores.columns = ['Name', 'Value']
    loading_scores2 = pd.Series(pca.components_[1], index=all_names)
    loading_scores2.columns = ['Name', 'Value']

    loading_scores.to_csv('./resources/user-pca1.csv')
    loading_scores2.to_csv('./resources/user-pcs2.csv')
    sorted_loading_scores = loading_scores.abs().sort_values(ascending=False)
    top_values = sorted_loading_scores[0:10].index.values
    print(loading_scores[top_values])
def parsehours(h):
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
if __name__ == "__main__":
    main()
    