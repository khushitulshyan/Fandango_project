from turtle import color
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# reading the csv file and hence exploring the dataframe.
fandango = pd.read_csv("fandango_scrape.csv")
print(fandango.head())
print("\n\n")
print(fandango.info())
print("\n\n")
print(fandango.describe())

# exploring the relationship between the popularity of film and its rating
plt.figure(figsize=(10,4),dpi=100)
sns.scatterplot(data=fandango,x='RATING',y='VOTES', color="black")
plt.show()

# Assuming that every film name has same format,i.e, FILM_NAME(year)
# fetching year from here and creating a new colummn for year 
fandango['YEAR'] = fandango['FILM'].apply(lambda title:title.split('(')[-1].replace(')',''))
print(fandango.head())

# counting the movies per year in a plot
sns.countplot(data=fandango,x='YEAR', color='black')
plt.show()

# 10 movies with highest rating
print(fandango.nlargest(10,'VOTES'))

# As we want to know the difference between movie numeric rating and visual star rating 
# we can remove movies with zero rating and explore that
fd_clean = fandango[fandango['VOTES'] > 0]
print((fd_clean['VOTES'] ==0).sum()) #to check if they are cleaned

plt.figure(figsize=(10,4),dpi=100)
sns.kdeplot(data=fd_clean,x='RATING',clip=[0,5],fill=True,label='True Rating')
sns.kdeplot(data=fd_clean,x='STARS',clip=[0,5],fill=True,label='Stars Displayed')

plt.legend(loc=(0.15,0.5))
plt.show()

fd_clean["STARS_DIFF"] = fd_clean['STARS'] - fd_clean['RATING'] 
fd_clean['STARS_DIFF'] = fd_clean['STARS_DIFF'].round(2)
print(fd_clean.head())

# count plot to see the difference
plt.figure(figsize=(12,4),dpi=100)
sns.countplot(data=fd_clean,x='STARS_DIFF',palette='magma')
plt.show()

# comparing the same with other websites
# exploring dataframe of other sites firstly
other_sites = pd.read_csv("all_sites_scores.csv")
print(other_sites.head())
print("\n\n")
print(other_sites.info())
print("\n\n")
print(other_sites.describe())

# scatterplot exploring the relationship between RT Critic reviews and RT User reviews.

plt.figure(figsize=(10,4),dpi=100)
sns.scatterplot(data=other_sites,x='RottenTomatoes',y='RottenTomatoes_User', color="black")
plt.xlim(0,100)
plt.ylim(0,100)
plt.show()

# Compare the critics ratings and the RT User ratings. 
# We will calculate this with RottenTomatoes-RottenTomatoes_User. 
# Note: Rotten_Diff here is Critics - User Score. So values closer to 0 means aggrement between Critics and Users. 
# Larger positive values means critics rated much higher than users. 
# Larger negative values means users rated much higher than critics.

other_sites['Rotten_Diff']  = other_sites['RottenTomatoes'] - other_sites['RottenTomatoes_User']
plt.figure(figsize=(10,4),dpi=100)
sns.histplot(data=other_sites,x='Rotten_Diff',kde=True,bins=25)
plt.title("RT Critics Score minus RT User Score")
plt.show()

# the top 5 movies users rated higher than critics on average
print("Users Love but Critics Hate")
print(other_sites.nsmallest(5,'Rotten_Diff')[['FILM','Rotten_Diff']])

# top 5 movies critics scores higher than users on average
print("Critics love, but Users Hate")
print(other_sites.nlargest(5,'Rotten_Diff')[['FILM','Rotten_Diff']])

# movie that has the highest IMDB user vote count
print(other_sites.nlargest(1,'IMDB_user_vote_count'))

# fandango as compared to other websites
df = pd.merge(fandango,other_sites,on='FILM',how='inner')
print(df.head())
# We need to *normalize* these values so they all fall between 0-5 stars and the relationship between reviews stays the same.
df['RT_Norm'] = np.round(df['RottenTomatoes']/20,1)
df['RTU_Norm'] =  np.round(df['RottenTomatoes_User']/20,1)
df['IMDB_Norm'] = np.round(df['IMDB']/2,1)
print(df.head())

norm_scores = df[['STARS','RATING','RT_Norm','RTU_Norm','IMDB_Norm']]
print(norm_scores.head())
print("\n\n")

# kdeplot for comparison
def move_legend(ax, new_loc, **kws):
    old_legend = ax.legend_
    handles = old_legend.legendHandles
    labels = [t.get_text() for t in old_legend.get_texts()]
    title = old_legend.get_title().get_text()
    ax.legend(handles, labels, loc=new_loc, title=title, **kws)

fig, ax = plt.subplots(figsize=(15,6),dpi=100)
sns.kdeplot(data=norm_scores,clip=[0,5],shade=True,palette='Set1',ax=ax)
move_legend(ax, "upper left")
plt.show()

# Clearly Fandango has an uneven distribution. We can also see that RT critics have the most uniform distribution.
# Let's directly compare these two. Compare the distribution of RT critic ratings against the STARS displayed by Fandango.
fig, ax = plt.subplots(figsize=(15,6),dpi=100)
sns.kdeplot(data=norm_scores[['RT_Norm','STARS']],clip=[0,5],shade=True,palette='Set1',ax=ax)
move_legend(ax, "upper left")
plt.show()

# Visualize the distribution of ratings across all sites for the top 10 worst movies.
norm_films = df[['STARS','RATING','RT_Norm','RTU_Norm','IMDB_Norm','FILM']]
print(norm_films.nsmallest(10,'RT_Norm'))
print('\n\n')
plt.figure(figsize=(15,6),dpi=100)
worst_films = norm_films.nsmallest(10,'RT_Norm').drop('FILM',axis=1)
sns.kdeplot(data=worst_films,clip=[0,5],shade=True,palette='Set1')
plt.title("Ratings for RT Critic's 10 Worst Reviewed Films")
plt.show()

# checking for the ratings we get
print(norm_films.iloc[25])
