# Kaggle - Job salary prediction competition

I competed in another Kaggle competition called [Job salary prediction](https://www.kaggle.com/c/job-salary-prediction). Goal of a competition was to predict salary based on job ads. It is a NLP competition. We have full text of Title, Full description and Location of ads. We know where in UK add was given and on which website.

Error metric is mean average error.

### Stats
I tried 291 combination of models, parameters and features.
I was doing this competition for 75 hours. I started in March.
I did 9 submissions on public leaderboard.

## Models

I used 2 different models:

* [Vowpal wabbit](https://github.com/JohnLangford/vowpal_wabbit)
* [Extra Tree regressor](http://scikit-learn.org/stable/modules/ensemble.html#extremely-randomized-trees) (ETr) from [sciki-learn](http://scikit-learn.org/stable/) Python library

I tried many other models but they didn't work so well. Random forest was little worse than Etr.


All models were doing log predictions of salaries unles speccialy noted.

## Features

For vowpall wabbit I used one model with all features. And another with locations changed in 5 parts. Same models as described [here](http://fastml.com/predicting-advertised-salaries/). Sadly I didn't try to improve score for Vowpall wabbit my score would probably be better.

For ETr I used different features:

1. 200 most frequent words in Title, FullDescription and LocationRaw
2. Same as first only [tf-idf](http://en.wikipedia.org/wiki/Tfidf) normalized values 
3. [Label encoded](http://scikit-learn.org/stable/modules/preprocessing.html#label-encoding) values in: Category, Contract Time, Contract Type

I also tried to get better features with [Gensim](http://radimrehurek.com/gensim/) but it didn't work out this time.

## Final model

Final model was average of 6 models:

* Vowpal wabbit with all features
* Vowpal wabbit with all features and location split in 5 parts
* Etr with 30 trees with features 1. and 3.
* Etr with 40 trees with features 1. and 3.
* Etr with 40 trees with features 1. and 3. with normal predictions (non log)
* Etr with 40 trees with features 2. and 3.

I chose this model because submodels gave best results in cross validation.
I was **26/285** on public leaderboard. And TBD on private.

## What I learned

* Cloud can be very usefull. (I used [picloud](https://www.picloud.com/))
* 32 bit computer even with PAE can not use 8 GB memory in scikit
* Make everything scriptable (I am getting better at this. But still it takes too long to change some paramters and run the same model on a test data and validation data. [Ramp](https://github.com/kvh/ramp) would probaly help but I didn't want to use it now because it uses pandas. I don't like pandas because it puts whole data in RAM and poor read speed. I saw too late that read speed is much better.
* Sleep is good ( better to sleep then work to 2-3 AM)


## Things to see into
* Ramp
* [Spearmint](http://www.cs.toronto.edu/~jasper/) (automatic parameter tuning)
* Vowpal wabbit

## Things I already do
* Train 60%, validation 20%, test split 20% from train set
* Cross validation
* Version control
