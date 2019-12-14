#Authors: William Carey - C16315253, Enda Keane - C16497656
#import libraries
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing
from sklearn.metrics import accuracy_score


#gethering the columns names
columnsName = open("headers.txt")
text_reader = columnsName.read()

#reading the csv file on which the predictions will be made
dataFrame = pd.read_csv('queries.csv',header = None, na_values = ["?", " ?", "? "])
dataFrame.columns = text_reader.split("\n")

#reading the file which we will train the model on
dataFrame2 = pd.read_csv('trainingset.csv',header = None, na_values = ["?", " ?", "? "])
dataFrame2.columns = text_reader.split("\n")

#gethering the converter to translate the catergorically features to continous
le = preprocessing.LabelEncoder()

#creating the collection which we do not want to train on
DictCollection = ["id"]



#Formatting the subscibed feature
dictFrameTarget = dataFrame2['subscribed'].to_dict()
for r in dictFrameTarget:
    if (dictFrameTarget[r] == "yes"):
        dictFrameTarget[r] = 1
    elif(dictFrameTarget[r] == "no"):
        dictFrameTarget[r] = 0


#converting the dataframe coloumns to continius features
ProcessedData = {}

le = preprocessing.LabelEncoder()
DictCollection = ["id","subscribed"]
for key in dataFrame2.columns:
    if key not in DictCollection:
        ProcessedData[key] = le.fit_transform(dataFrame2[key])
        

ProcessedPrediction = {}

for key in dataFrame.columns:
    if key not in DictCollection:
        ProcessedPrediction[key] = le.fit_transform(dataFrame[key])
        

#converting the dict to dataframe
df = pd.DataFrame(ProcessedData) 
predDf = pd.DataFrame(ProcessedPrediction)

#training the dataset with an appropriate amount of training data.
train,test,train_labels,test_labels = train_test_split(df,
                                                       dictFrameTarget,
                                                       test_size = 0.45,
                                                       random_state=42)

#Creating the model and fitting it to the data
gnb = GaussianNB()
model = gnb.fit(train, train_labels)

#Make a prediction and test accuracy
preds = gnb.predict(predDf)

#changing the numeric representation of the predictions to string representation
predsAns = []
for i in preds:
    if i == 0:
        predsAns.append("no")
    elif i == 1:
        predsAns.append("yes")
        
dataPreds = pd.DataFrame(predsAns)

#Old code for testing accuracy against the left over training set
#preds = gnb.predict(test)
#print(preds)
#print(accuracy_score(test_labels, preds))
#Result: 80%

#concat the two dataframes and write to csv file
compPredictions = pd.concat([dataFrame["id"], dataPreds],axis=1)

compPredictions.to_csv('PredictionFile.csv')