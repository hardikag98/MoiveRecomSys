# MoiveRecomSys
Create a movie recommendation system using latent factor models, neighbourhood methods and content based methods on MovieLens data

This project experiments with several different approaches (Content-based methods, neighborhood methods and latent factor models) to build a movie recommender system based on data from MovieLens. The recommender systems are trained on a set of movie ratings (and on metadata ofthe movies in cases of content based methods) by different users and is then used to predict how these users will like movies which they have not yet seen/rated by predicting what the users will rate these movies. All approaches are evaluated on their predicted ratings. The evaluation metric used is Root Mean Square Error (RMSE).

DATA DESCRIPTION:
We use ratings data from MovieLens and corresponding metadata about the movies present in the MovieLens dataset has been taken from TMDB (The Movies DataBase).

The data is contained within 2 different files:
 movies_metadata: This file contains metadata about the movies such as production country, cast, overview, tagline, revenue, runtime, etc. However, for the purpose of this project only ‘overview’, ’tagline’ ‘genre’, ’keywords’ and ‘cast’ is used.
 ratings: This file contains all the user ratings. Ratings are on a scale of 0.5-5 with multiples of 0.5. The columns in this file are: [userId, movieId, rating].

METHODOLOGY

Collaborative Filtering:
Collaborative filtering is an approach to building recommender systems that analyzes relationships between users and interdependencies among products to identify new user-item associations. There are broadly speaking, 2 types of collaborative filtering - neighborhood methods and latent factor models.

1) Neighborhood Methods
Neighborhood methods are centered on computing the relationships between items or, alternatively, between users.

![image](https://user-images.githubusercontent.com/59964344/138456863-904221cc-e12a-4083-a1fb-abc73a7071a6.png)


1.1) Item oriented neighborhood filtering
The item oriented neighborhood approach evaluates a user’s preference for an item based on ratings of neighboring items by the same user. A product’s neighbors are other products that tend to get similar ratings when rated by the same user. To predict a particular user’s rating for some movie X, we would look for the movies nearest neighbors that this user actually rated. Predicted rating is calculated as below.

![image](https://user-images.githubusercontent.com/59964344/138456923-cd9ebf64-a488-459c-8a4e-73cc467f2d11.png)

where bui = μ + bi + bu , bui is the bias involved in rating rui and accounts for the user and item effects. The overall average rating is denoted by μ; the parameters bu and bi indicate the observed deviations of user u and item i, respectively, from the average.

To evaluate similarity between items, the metric used was ‘pearson_baseline’ which is given by:

![image](https://user-images.githubusercontent.com/59964344/138457616-6f7d263f-83f0-4337-8b16-e0636cb7a0f1.png)

1.2)	User oriented neighborhood filtering 
This approach is similar to item oriented neighborhood filtering except that instead of evaluating a user’s preference for an item based on rating of neighboring items by the same user, it uses neighboring users on the same item. A user’s neighbor are other neighbors that tend to give similar ratings when rating the same item. To predict a particular user’s rating for some movie X, we would look for the user’s nearest neighbors that have rated movie X. 

![image](https://user-images.githubusercontent.com/59964344/138457647-ea7bce31-258c-44d1-a1a2-7b3a568bbc62.png)


2) Latent Factor Models 
Latent factor models are an alternative approach that tries to explain ratings by characterizing both items and users on factors inferred from the ratings patterns.

![image](https://user-images.githubusercontent.com/59964344/138457662-183b8956-74c6-47bd-8010-cd27b34a967a.png)

Using the concept of Matrix Factorization, we assume the existence of d latent features such that our U x I (No.of Users x No.of Items) rating matrix R can be represented as the product of two lower-dimension matrices: Q of size U x d and P of size d x I. More specifically, the predicted rating from User u on Movie i can be represented as:
〖 r ̂〗_ui= μ+b_i+b_u+ q_i^T p_u
	The challenge behind Matrix Factorization comes down to estimating the latent features. As we want to optimize the RMSE between actual and predicted ratings with regards to Q and P, we obtain the following Loss function to minimize, including regularization terms for both matrices:

![image](https://user-images.githubusercontent.com/59964344/138457714-7f2eba31-45d6-46b9-a2c0-6a0dfe584a9b.png)


Stochastic Gradient Descent is used to minimize the above equation and hence obtain P,Q, b_i  and b_u.

Content Based Recommender Systems
One popular technique of recommender systems is content-based filtering. The idea in content-based filtering is to tag products using certain keywords or describe them by an overview of their appearance/features/content thereby creating a sort of profile for each product. We then understand what the user likes and look up different products with similar profiles/attributes in the database and recommend these to the user.In our case, we want to predict the rating for some unseen movie X by an user for which we shall use the following methodology:
First, we represent our dataset of movies as a TF-IDF matrix F of size I x W where W is the number of features or terms and I is the number of movies (Only top 1000 features/terms, after removing the stop words, were used in this case). 
TF-IDF is an information retrieval technique to estimate the importance of a word w appearing in a snippet of text i. It combines Term Frequency (tf as # of times a word appears in a snippet divided by total words) with Inverse Document Frequency (idfw = log(I/dfw) where dfw is the number of snippets containing w , so the more frequent across snippets, the lower the score): 

F_iw=〖tf〗_iw* 〖idf〗_w  
From there, we compute an affinity factor between movies i and j in the form of Cosine Similarity s and obtain a I x I Affinity Matrix. We then use these similarities as weights to predict user ratings with the idea that if movie X was rated 5 by a specific user and is very similar to Y, chances are the user will rate it high as well. [Source: Building the optimal Book Recommender and measuring the role of Book Covers in predicting user ratings by Cécile Logé and Alexander Yoffe]
![image](https://user-images.githubusercontent.com/59964344/138457813-3b44679f-f1ee-4eb7-99d8-5b4a6e959cee.png)

〖 r ̂〗_ui=b_(ui )+  (∑_(〖v ∈ I〗_u)▒〖〖(r〗_uv-b_ui)〗 s_iv)/(∑_(〖v ∈ I〗_u)▒s_iv )
where Iv is the list of movies user u already rated, meaning predicted ratings are weighted averages of known ratings from the training set.
This methodology was implemented using the ‘description’ column of our metadata which contains the ‘overview’ (brief textual description/summary of the movie) plus the ‘tagline’ column. Another model with a slightly different methodology was implemented using the ‘genre’, ’keywords’ and ‘cast’ column (which were combined to create the ‘metadata’ column) where instead of tf-idf the data was one hot encoded, i.e., a separate columns was created for each genre and cast member (which appeared more than once in the data) and the value under those columns (0/1) represented if that genre/cast member was a feature of that particular movie. Following this the same methodology of computing an affinity factor between movies in form of cosine similarity and then using that predict ratings was used.
