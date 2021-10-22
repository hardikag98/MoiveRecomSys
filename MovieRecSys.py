#Importing the required packages 
import pandas as pd
import numpy as np
import io
import matplotlib as plt
from surprise import Reader, Dataset, SVD
from surprise import AlgoBase, PredictionImpossible
from surprise.model_selection import cross_validate,GridSearchCV, train_test_split, KFold
from surprise import accuracy
from surprise import KNNBaseline, BaselineOnly
import os
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from sklearn.metrics import mean_squared_error

#Importing data
path = os.getcwd()
ratings = pd.read_csv(path+"/ratingsnew.csv")
smd = pd.read_csv(path+"/smd.csv")
ratings = ratings.drop(['Unnamed: 0'],axis=1)
smd=smd.drop(['Unnamed: 0'],axis=1)
smd['tagline'] = smd['tagline'].fillna('')
smd['description'] = smd['overview'] + smd['tagline']
smd['description'] = smd['description'].fillna('')
smd['metadata'] = smd['metadata'].fillna('')
smd['desc&meta']=smd['description']+smd['metadata']

#EDA
##Rating distribution
plt.hist(ratings.rating,bins=10,range=[0.5,5.5])
plt.title('Ratings distribution')
plt.xlabel("Rating")
plt.ylabel("Count")
plt.show()

## Distribution of number of ratings per user
plt.boxplot(ratings.groupby("userId").count().rating,labels=[''])
plt.title("Boxplot of number of ratings per user")
plt.yscale("log")
plt.xlabel("")
plt.show()

## Distribution of number of ratings per movie
moviesdata = ratings.groupby("movieId").count().rating
plt.boxplot(moviesdata,labels=[''])
plt.yscale("log")
plt.title('Boxplot of number of ratings per movie')
plt.show()

#Setting seed to ensure reproducible results
np.random.seed(42)

# Splitting data into train and test set 
ratings = ratings[['userId', 'movieId', 'rating']]
sratings = ratings.sample(frac=1, random_state=42)
traindata = sratings[:int(0.9*len(sratings))][['userId', 'movieId', 'rating']]
testdata = sratings[int(0.9*len(sratings)):][['userId', 'movieId', 'rating']]

#Loading data into surprise package 
reader = Reader()
trainingdata = Dataset.load_from_df(traindata[['userId', 'movieId', 'rating']], reader)
fulldata = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
trainset, testset = train_test_split(fulldata, random_state=42,test_size=0.1)

#Train and validation RMSE for SVD algorithm 
np.random.seed(42)
svd = SVD(n_factors=100,reg_all=0.1,n_epochs=25,random_state=42,lr_all=0.02)
cross_validate(svd,trainingdata,cv=5,return_train_measures=True,n_jobs=-1,verbose=True)

#Test set RMSE for SVD algorithm 
algo = SVD(n_factors=100,reg_all=0.1,n_epochs=25,random_state=42,lr_all=0.02)
algo.fit(trainset)
predictions1 = algo.test(testset)
accuracy.rmse(predictions1)

#Train and validation RMSE for user oriented neighborhood method 
np.random.seed(42)
sim_options = {'name':'pearson_baseline','user_based':True}
bopt = {'method': 'als'}
knn = KNNBaseline(k=25,min_k=7,sim_options=sim_options,bsl_options=bopt)
cross_validate(knn,trainingdata,cv=5,return_train_measures=True,n_jobs=-1,verbose=True)

#Test set RMSE for user oriented neighborhood method
sim_options = {'name':'pearson_baseline','user_based':True}
bopt = {'method': 'als'}
algo = KNNBaseline(k=15,min_k=7,sim_options=sim_options,bsl_options=bopt)
algo.fit(trainset)
predictions2 = algo.test(testset)
accuracy.rmse(predictions2)

#Train and validation RMSE for item oriented neighborhood method 
np.random.seed(42)
sim_options = {'name':'pearson_baseline','user_based':False}
knn = KNNBaseline(k=20,min_k=7,sim_options=sim_options)
cross_validate(knn,trainingdata,cv=5,return_train_measures=True,n_jobs=-1,verbose=True)

#Test set RMSE for item oriented neighborhood method
sim_options = {'name':'pearson_baseline','user_based':False}
algo = KNNBaseline(k=20,min_k=7,sim_options=sim_options)
algo.fit(trainset)
predictions3 = algo.test(testset)
accuracy.rmse(predictions3)

#Content Based recommender systems:
#calculate tfidf vector
tf = TfidfVectorizer(stop_words='english',strip_accents='unicode',max_features=1000)
tfidf_matrix = tf.fit_transform(smd.description)

#calculate cosine similarity
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

#Creating a function to use content based recommeder system to predict ratings 
def contentrec(userid,itemid,cosinesim,traindata,testdata,algo,k=40):
    rated = traindata[traindata.userId==userid][['movieId','newrating']]
    bias = algo.predict(userid,itemid)[3]
    a=smd[smd.movieId.isin(rated['movieId'])].index
    itemindex = smd[smd.movieId==itemid].index[0]
    similarity = cosinesim[itemindex,list(a)]
    rank = np.array(similarity).argsort()[-k:]
    ratings = np.array(rated['newrating'])[rank]
    similarity = similarity[rank]
    if (np.sum(similarity)!=0):
        pred = bias + np.sum(similarity*ratings)/np.sum(similarity)
    else:
        pred = bias 
    return pred

def itemaffinity(traindata,testdata,cosinesim):   
    reader = Reader()
    data = Dataset.load_from_df(traindata[['userId', 'movieId', 'rating']], reader)
    trainset = data.build_full_trainset()
    algo = BaselineOnly()
    algo.fit(trainset)
    
    traindata['baseline']=0
    for index, i in traindata.iterrows():
        traindata.loc[index,'baseline'] =  algo.predict(i['userId'],i['movieId'])[3]
    traindata['newrating'] = traindata['rating'] - traindata['baseline']

    pred_rating=np.array([])
    for index,i in testdata.iterrows():
        pred = contentrec(int(i['userId']),int(i['movieId']),cosinesim,traindata,testdata,algo)
        pred_rating = np.append(pred_rating,pred)
    
    pred_rating1=np.array([])
    for index,i in traindata.iterrows():
        pred = contentrec(int(i['userId']),int(i['movieId']),cosinesim,traindata,testdata,algo)
        pred_rating1 = np.append(pred_rating1,pred)
        
    return (pred_rating,pred_rating1)

#Train and validation RMSE for content based(desc tf-idf) model 
from  sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
kf = KFold(random_state=42,shuffle=True)
CVerrors = []
trainerrors=[]
for train_index, test_index in kf.split(traindata):
    traindata1 = traindata.iloc[train_index,:]
    testdata1 = traindata.iloc[test_index,:]

    (pred1,pred2) = itemaffinity(traindata1,testdata1,cosine_sim)
    cverror = mean_squared_error(testdata1.rating,pred1,squared=False)
    trainerror = mean_squared_error(traindata1.rating,pred2,squared=False)
    CVerrors.append(cverror)
    trainerrors.append(trainerror)
print(CVerrors)
print(np.mean(CVerrors))
print(trainerrors)
print(np.mean(trainerrors))

#Test set RMSE for content based model 
(pred1,pred2) = itemaffinity(traindata,testdata,cosine_sim)
mean_squared_error(testdata.rating,pred1,squared=False)

#Creating one hot encoded vector of genre, cast and keywordss
countvec = TfidfVectorizer(stop_words='english',strip_accents='unicode',max_df=0.9,min_df=3)
countvec_matrix = countvec.fit_transform(smd.metadata)
cosine_sim1 = cosine_similarity(countvec_matrix,countvec_matrix)

#Train and validation RMSE for cont based(metadata one hot encoded) RS
from  sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
kf = KFold(random_state=42,shuffle=True)
CVerrors = []
trainerrors=[]
for train_index, test_index in kf.split(traindata):
    traindata1 = traindata.iloc[train_index,:]
    testdata1 = traindata.iloc[test_index,:]
    
    (pred1,pred2) = itemaffinity(traindata1,testdata1,cosine_sim1)
    cverror = mean_squared_error(testdata1.rating,pred1,squared=False)
    trainerror = mean_squared_error(traindata1.rating,pred2,squared=False)
    CVerrors.append(cverror)
    trainerrors.append(trainerror)
print(CVerrors)
print(np.mean(CVerrors))
print(trainerrors)
print(np.mean(trainerrors))

#Test set RMSE for cont based(metadata one hot encoded) RS
(pred1,pred2) = itemaffinity(traindata,testdata,cosine_sim1)
mean_squared_error(testdata.rating,pred1,squared=False)

#Function for ensemble model 
w1=0.35
w2=0.35
w3=0.3
class MyOwnAlgorithm(AlgoBase):

    def __init__(self, sim_options1={'name':'pearson_baseline','user_based':False},
                 sim_options2={'name':'pearson_baseline','user_based':True},min_k=7,
                 bsl_options = {},#'method': 'als', 'n_epochs': 10, 'reg_u': 3, 'reg_i': 2}, 
                k1=20,k2=15,n_factors=100,reg_all=0.1,n_epochs=25,lr_all=0.02,
                random_state=42):

        AlgoBase.__init__(self, sim_options=sim_options1,bsl_options=bsl_options)
        
        knnitem = KNNBaseline(sim_options=sim_options1,k=k1,min_k=7)
        knnuser = KNNBaseline(sim_options=sim_options2,k=k2,min_k=7)
        svd = SVD(n_factors=n_factors,reg_all=reg_all,n_epochs=n_epochs,
                  lr_all=lr_all,random_state=random_state)
        
        self.knnitem = knnitem
        self.knnuser = knnuser
        self.svd = svd
    
    def fit(self, trainset):
        
        AlgoBase.fit(self, trainset)
                
        self.knnitemparam = self.knnitem.fit(trainset)   
        self.knnuserparam = self.knnuser.fit(trainset)     
        self.svdparam = self.svd.fit(trainset)
        
        return self
    
    def estimate(self, u, i):
        
        svdest = self.svdparam.estimate(u,i)
        knnitemest = self.knnitemparam.estimate(u,i)
        knnuserest = self.knnuserparam.estimate(u,i)

        try:
            ensembleest = w1*np.array(svdest) + w2*np.array(knnitemest[0]) + w3*np.array(knnuserest[0])
        except IndexError:
            try:
                ensembleest = w1*np.array(svdest) + w2*np.array(knnitemest[0]) + w3*np.array(knnuserest)
            except IndexError:
                try:
                    ensembleest = w1*np.array(svdest) + w2*np.array(knnitemest) + w3*np.array(knnuserest[0])
                except IndexError:
                    ensembleest = w1*np.array(svdest) + w2*np.array(knnitemest) + w3*np.array(knnuserest)

        return ensembleest

#Train and Validation RMSE for ensemble model 
from surprise.model_selection import KFold
algo = MyOwnAlgorithm()        
kfold = KFold(random_state=42) 
cross_validate(algo,trainingdata,cv=kfold,return_train_measures=True,verbose=True,n_jobs=-1)

#Test set RMSE for ensemble model
algo = MyOwnAlgorithm()  
algo.fit(trainset)
predictions = algo.test(testset)
accuracy.rmse(predictions)

