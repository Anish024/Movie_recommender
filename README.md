# Movie_recommender
this project recommends movie on your preselected options from given pool using ML 

A recommender system is an information filtering system that seeks to predicts the rating given by a user to an item. This predicted rating then used to recommend items to the user. The item for which the predicted rating is high will be recommended to the user. This recommender system is utilized in recommendation of a broad range of items. For instance, it can be used to recommend movies, products, videos, music, books, news, Facebook friends, clothes, Twitter pages, Android/ios apps, hotels, restaurants, routes etc. It is used by almost all of the major companies to enhance their business and to enrich user experience like YouTube for recommending videos, Amazon & Ebay for recommending products, Netflix for recommending Movies, Airbnb for recommending rooms and hotels, Facebook for recommending friends etc.

This USER-ITEM matrix is very sparse matrix, which means that many cells in this matrix are empty. Since, there are many items, and a single user cannot give rating to all of the items. In real-world, a single user does not give ratings to even 1% of the total items. Therefore, around 99% of the cells of this matrix are empty. These empty cells can be represented by “NaN” means not a number. For example, let say ’n’ is 1-Million and ‘m’ is 10k. Now n*m is 10 ^10 which is a very large number. Now, let say an average user gives rating to 5 items. Then total number of ratings given on an average will be 5 * 1-Million = 5 * 10⁶ ratings. Now there is a metric called Sparsity of a matrix.

Sparsity of Matrix = Number of Empty cells / Total Number of cells.

Here, Sparsity of matrix = (10¹⁰ — 5*10⁶) / 10¹⁰ = 0.9995
It means that 99.95% of cells are empty. This is extreme sparsity actually.

Task of Recommender System(RS): Let say, if there is a user Ui who likes item I1, I5, I7. Then we have to recommend user Ui an item such Ij which he/she will most probably like.
ontent based filtering is similar in approach with classical machine learning techniques. It needs a way to represent an item Ij and a user Ui. Here, we need to collect information about an item Ij and a user Ui then finally we need to create features of both user Ui and Ij. Then we combine those features and feed them to a Machine Learning model for training. Here, label will be Aij, which is the corresponding rating given by a user Ui on item Ij.

Let’s take an example to understand this in more detail:

Let say the item in our data-set is a movie. Now we can create its feature like this:
· Genre of Movie

· Year of Release

· Lead Actor

· Director

· Box Office Collection

· Budget

Similarly, features for user can also be created:

· Likes and dislikes of users

· Gender of a user

· Age of user

· Where user lives

As soon as we have the above mentioned information about items and users, we can create an item vector which shall contain information about the item which is mentioned above. Then, we can similarly create a user vector which shall contain information about the user which is mentioned above. We can generate features for each user Ui and an item Ij. Finally we can combine these features and create a big data-set which can be suitable for feeding to Machine Learning model.

Here, above I have just explained an approximate way to create features for content based filtering. These features should be carefully designed so that they impact the rating/label directly without being dependent on each other. It is always better to create as independent features as possible and at the same time they should be very much dependent on rating/label means that they should directly affect the rating/label.




## Team Name
Anish Agrawal 17ucs024
Shatishay Jain 17ucs146
Tarun Gupta  17ucs168
