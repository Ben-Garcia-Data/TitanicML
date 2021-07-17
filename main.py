# We have an input dataset
# We use input variables to calculate a likihood of surviving and make like 100
# different ways to calulate liklihood each time.
# We test each of the methods against the true values (dead or alive), score & rank.
# Take rando distribution weighted to the top of scores

import pandas
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pandas.read_csv("train.csv")
df2 = pandas.read_csv("test.csv")

# print(df.head())

def clean_data(input, has_survival):

    print("Cleaning data")

    replacement_dict = {}
    replacement_dict['male'] = 1
    replacement_dict['female'] = 2
    replacement_dict['S'] = 1
    replacement_dict['C'] = 2
    replacement_dict['Q'] = 3
    if has_survival:
        correct = list(input['Survived'])
        input = input.drop(columns = ['Survived'])

    input = input.drop(columns = ['Name', 'Ticket','Cabin']) #Am reluctant to drop cabin data but too many NaN & I don't know what each value means

    input['Fare'] = input['Fare'].replace(np.NaN, input['Fare'].median())  # Replace missing values with mean
    input['Age'] = input['Age'].replace(np.NaN, input['Age'].mean()) # Replace missing values with mean
    input['Embarked'] = input['Embarked'].replace(np.NaN, input['Embarked'].mode()[0]) # Replace missing values with mode


    input = input.replace(replacement_dict)

    if has_survival:
        return correct, input
    else:
        return input

correct, df = clean_data(df, True)

"""
Data Dictionary
Variable	Definition	Key
survival	Survival	0 = No, 1 = Yes
pclass	Ticket class	1 = 1st, 2 = 2nd, 3 = 3rd
sex	Sex	
Age	Age in years	
sibsp	# of siblings / spouses aboard the Titanic	
parch	# of parents / children aboard the Titanic	
ticket	Ticket number	
fare	Passenger fare	
cabin	Cabin number	
embarked	Port of Embarkation	C = Cherbourg, Q = Queenstown, S = Southampton
"""

class passenger():
    def __init__(self, Id, Survived, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked):
        self.Id = Id
        self.Survived = Survived # Test against this.
        self.Pclass = Pclass
        self.Name = Name
        self.Sex = Sex
        self.Age = Age
        self.SibSp = SibSp
        self.Parch = Parch
        self.Ticket = Ticket
        self.Fare = Fare
        self.Cabin = Cabin
        self.Embarked = Embarked

print("Making dataset")
dataset = df.values.tolist()

print("Making pipe")
pipe = make_pipeline(
    StandardScaler(),
    LogisticRegression()
)

print("Splitting data")
X, y = [dataset,correct]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

print("Fitting")
pipe.fit(X_train, y_train)

print("Checking accuracy")
acc = accuracy_score(pipe.predict(X_test), y_test)

print(acc)


df2 = clean_data(df2, False)

# print(df2)
# print("\n")
# print(df2.dtypes)

res = pipe.predict(df2)

df3 = pandas.DataFrame({'PassengerId': df2['PassengerId'],'Survived': res})
print(df3)
df3.to_csv("Output.csv",index=False)


