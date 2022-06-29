import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
from datetime import datetime
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from collections import Counter
def main():
    sample = pd.read_csv("./resources/test.csv")
    plotAttributeCount(sample)
    plotReviewCountStars(sample)
    plotStarOccurrence(sample)
    pca(sample)

def plotStarOccurrence(sample):
    all_stars = sample['stars']
    plt.rcParams['font.size'] = '16'
    ratings = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
    star_occurrences = {key: 0 for key in ratings}
    for stars in all_stars:
        star_occurrences[stars] += 1
    plt.plot(star_occurrences.keys(), star_occurrences.values(), '-ok')
    plt.savefig('./resources/Star_Occurrence.png')
    plt.show()
def plotReviewCountStars(sample):
    all_reviewcount = sample['review_count']
    all_stars = sample['stars']
    plt.scatter(all_stars, all_reviewcount)
    plt.xlabel('Star rating')
    plt.ylabel('# of Reviews')
    plt.savefig('./resources/Review_Count.png')
    plt.show()
def plotAttributeCount(sample):
    all_attributes = sample['attributes']
    all_stars = sample['stars']
    attributes_nums = {}
    i = 0
    for attr in all_attributes:
        if (isfloat(attr)):
            attributes_nums[i] = 0
        else:
            attributes_nums[i] = attr.count("True")
        i+=1
    c = Counter(zip(all_stars.values,attributes_nums.values()))
    s = [10*c[x1,y1] for x1,y1 in zip(all_stars.values,attributes_nums.values())]
    plt.scatter(all_stars, attributes_nums.values(), s=s)
    plt.xlabel("Star Rating")
    plt.ylabel("# of Attributes")
    plt.show()
def isfloat(num):
    try:
        float(num)
        return True
    except ValueError:
        return False
# PCA 
def pca(sample):
    all_attributes = sample['attributes']
    all_hours = sample['hours']
    print(sample.head())
    all_names = sample['name']
    total_time = 0
    all_total_times = {}
    j = 0
    for time in all_hours:
        if isfloat(time):
            all_total_times[j] = 0
            j+=1
            continue
        business = json.loads(time.replace('\'', '\"'))
        for day in business:
            times = business[day].split("-")
            time1 = datetime.strptime(times[0], "%H:%M")
            time2 = datetime.strptime(times[1], "%H:%M")
            total_time += (time2 - time1).seconds/3600
        all_total_times[j] = total_time
        total_time = 0
        j+=1
    attributes_nums = {}
    i = 0
    for attr in all_attributes:
        if (isfloat(attr)):
            attributes_nums[i] = 0
        else:
            attributes_nums[i] = attr.count("True")
        i+=1
    pcaDF = sample[['stars', 'is_open', 'review_count']]
    pcaDF.index = all_names
    pcaDF2 = pd.DataFrame({'attribute_count': attributes_nums, 'weekly_hours': all_total_times})
    pcaDF2.index = all_names
    pcaFrames = [pcaDF, pcaDF2]
    pcaData = pd.concat(pcaFrames, axis=1, join='outer')
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
    plt.savefig('./resources/PCA_Figure_1.png')
    plt.show()
    pca_df = pd.DataFrame(pca_data, index=[pcaData[['stars', 'is_open', 'review_count', 'attribute_count', 'weekly_hours']]], columns=labels)
    plt.scatter(pca_df.PC1, pca_df.PC2)
    plt.xlabel('PC1 - {0}%'.format(per_var[0]))
    plt.ylabel('PC2 - {0}%'.format(per_var[1]))
    for sample in pca_df.index:
        plt.annotate(sample, (pca_df.PC1.loc[sample], pca_df.PC2.loc[sample]))
    plt.savefig('./resources/PCA_Figure_2.png')
    plt.show()

    loading_scores = pd.Series(pca.components_[0], index=all_names)
    loading_scores.columns = ['Name', 'Value']
    loading_scores2 = pd.Series(pca.components_[1], index=all_names)
    loading_scores2.columns = ['Name', 'Value']

    loading_scores.to_csv('./pca1.csv')
    loading_scores2.to_csv('./pcs2.csv')
    sorted_loading_scores = loading_scores.abs().sort_values(ascending=False)
    top_values = sorted_loading_scores[0:10].index.values
    print(loading_scores[top_values])
    

if __name__ == "__main__":
    main()
    