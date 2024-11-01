import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler



def transform_gender(x):
    return 1 if x == 'male' else 0

class Data_Preprocessing:
    def __init__(self, feature1, feature2, Class1, Class2, file_path="birds.csv"):
        self.feature1 = feature1
        self.feature2 = feature2
        self.Class1 = Class1
        self.Class2 = Class2
        self.file_path = file_path

    def preprocessing(self):
        # Load dataset
        data = pd.read_csv(self.file_path)
        #print("Initial data loaded:")
        #print(data.head())

        data["gender"] = data["gender"].fillna(data["gender"].mode()[0])
        # Convert gender to numeric
        data["gender"] = data["gender"].apply(transform_gender)
        
        
        # Standardization
        scaler = StandardScaler()
        standardized_columns = ['body_mass', 'beak_length', 'beak_depth', 'fin_length']
        data[standardized_columns] = scaler.fit_transform(data[standardized_columns])

        Selected_data = data[[self.feature1, self.feature2, 'bird category']]
        Selected_data = Selected_data[Selected_data['bird category'].isin([self.Class1, self.Class2])]
        Selected_data['bird category'] = Selected_data['bird category'].map({self.Class1: 1, self.Class2: -1})

        class1_data = Selected_data[Selected_data['bird category'] == 1]
        class2_data = Selected_data[Selected_data['bird category'] == -1]

        test_size_class = int(len(class1_data) * 0.4)
        train_size_class = len(class1_data) - test_size_class

        class1_train = class1_data.sample(n=train_size_class, random_state=1)
        class1_test = class1_data.drop(class1_train.index)

        class2_train = class2_data.sample(n=train_size_class, random_state=1)
        class2_test = class2_data.drop(class2_train.index)
     
        self.X_train = pd.concat([class1_train[[self.feature1, self.feature2]], class2_train[[self.feature1, self.feature2]]])
        self.y_train = pd.concat([class1_train['bird category'], class2_train['bird category']])
        self.X_test = pd.concat([class1_test[[self.feature1, self.feature2]], class2_test[[self.feature1, self.feature2]]])
        self.y_test = pd.concat([class1_test['bird category'], class2_test['bird category']])

        print("\nTraining data (X_train, y_train):")
        #print(self.X_train.head(), self.y_train.head())
        print("Training data shape:", self.X_train.shape, self.y_train.shape)
       
        print("\nTesting data (X_test, y_test):")
        #print(self.X_test.head(), self.y_test.head())
        print("Testing data shape:", self.X_test.shape, self.y_test.shape)

    def splitdata(self):
        return self.X_train, self.X_test, self.y_train, self.y_test