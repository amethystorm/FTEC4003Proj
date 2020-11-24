import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import csv


this_folder = os.path.dirname(os.path.abspath(__file__))
file_name = this_folder+'/train.csv'

# Use only the last 11 columns
useful_cols = list(pd.read_csv(file_name, nrows =1))
dataframe = pd.read_csv(file_name,usecols=useful_cols[-11:])
# print(useful_cols[-11:])
# Split train & val datasets
val_dataframe = dataframe.sample(frac=0.2, random_state=1337)
train_dataframe = dataframe.drop(val_dataframe.index)

# print(dataframe.shape)
# print(
#     "Using %d samples for training and %d for validation"
#     % (len(train_dataframe), len(val_dataframe))
# )

# generate tensorflow dataset object
def dataframe_to_dataset(dataframe):
    dataframe = dataframe.copy()
    labels = dataframe.pop("Exited")
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    ds = ds.shuffle(buffer_size=len(dataframe))
    return ds


train_ds = dataframe_to_dataset(train_dataframe)
val_ds = dataframe_to_dataset(val_dataframe)

# for x, y in train_ds.take(1):
#     print("Input:", x)
#     print("Target:", y)

# Batch the datasets
train_ds = train_ds.batch(32)
val_ds = val_ds.batch(32)

from tensorflow.keras.layers.experimental.preprocessing import Normalization
from tensorflow.keras.layers.experimental.preprocessing import CategoryEncoding
from tensorflow.keras.layers.experimental.preprocessing import StringLookup


def encode_numerical_feature(feature, name, dataset):
    # Create a Normalization layer for our feature
    normalizer = Normalization()

    # Prepare a Dataset that only yields our feature
    feature_ds = dataset.map(lambda x, y: x[name])
    feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))

    # Learn the statistics of the data
    normalizer.adapt(feature_ds)

    # Normalize the input feature
    encoded_feature = normalizer(feature)
    return encoded_feature


def encode_string_categorical_feature(feature, name, dataset):
    # Create a StringLookup layer which will turn strings into integer indices
    index = StringLookup()

    # Prepare a Dataset that only yields our feature
    feature_ds = dataset.map(lambda x, y: x[name])
    feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))

    # Learn the set of possible string values and assign them a fixed integer index
    index.adapt(feature_ds)

    # Turn the string input into integer indices
    encoded_feature = index(feature)

    # Create a CategoryEncoding for our integer indices
    encoder = CategoryEncoding(output_mode="binary")

    # Prepare a dataset of indices
    feature_ds = feature_ds.map(index)

    # Learn the space of possible indices
    encoder.adapt(feature_ds)

    # Apply one-hot encoding to our indices
    encoded_feature = encoder(encoded_feature)
    return encoded_feature


def encode_integer_categorical_feature(feature, name, dataset):
    # Create a CategoryEncoding for our integer indices
    encoder = CategoryEncoding(output_mode="binary")

    # Prepare a Dataset that only yields our feature
    feature_ds = dataset.map(lambda x, y: x[name])
    feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))

    # Learn the space of possible indices
    encoder.adapt(feature_ds)

    # Apply one-hot encoding to our indices
    encoded_feature = encoder(feature)
    return encoded_feature


# Categorical features encoded as integers
has_credit_card  = keras.Input(shape=(1,), name="HasCrCard", dtype="int64")
is_active_member = keras.Input(shape=(1,), name="IsActiveMember", dtype="int64")

# Categorical feature encoded as string
geography = keras.Input(shape=(1,), name="Geography", dtype="string")
gender = keras.Input(shape=(1,), name="Gender", dtype="string")

# Numerical features
credit_score = keras.Input(shape=(1,), name="CreditScore")
age = keras.Input(shape=(1,), name="Age")
tenure = keras.Input(shape=(1,), name="Tenure")
balance = keras.Input(shape=(1,), name="Balance")
num_of_products = keras.Input(shape=(1,), name="NumOfProducts")
estimated_salary = keras.Input(shape=(1,), name="EstimatedSalary")

all_inputs = [
    has_credit_card,
    is_active_member,
    geography,
    gender,
    credit_score,
    age,
    tenure,
    balance,
    num_of_products,
    estimated_salary,
]

# Integer categorical features
has_credit_card_encoded = encode_integer_categorical_feature(has_credit_card,"HasCrCard",train_ds)
is_active_member_encoded = encode_integer_categorical_feature(is_active_member,"IsActiveMember",train_ds)

# String categorical features
geography_encoded = encode_string_categorical_feature(geography,"Geography",train_ds)
gender_encoded = encode_string_categorical_feature(gender,"Gender",train_ds)

# Numerical features
credit_score_encoded = encode_numerical_feature(credit_score,"CreditScore",train_ds)
age_encoded = encode_numerical_feature(age,"Age",train_ds)
tenure_encoded = encode_numerical_feature(tenure,"Tenure",train_ds)
balance_encoded = encode_numerical_feature(balance,"Balance",train_ds)
num_of_products_encoded = encode_numerical_feature(num_of_products,"NumOfProducts",train_ds)
estimated_salary_encoded = encode_numerical_feature(estimated_salary,"EstimatedSalary",train_ds)

all_features = layers.concatenate(
    [
        has_credit_card_encoded,
        is_active_member_encoded,
        geography_encoded,
        gender_encoded,
        credit_score_encoded,
        age_encoded,
        tenure_encoded,
        balance_encoded,
        num_of_products_encoded,
        estimated_salary_encoded,
    ]
)

x = layers.Dense(50, activation="relu")(all_features)
x = layers.Dense(25, activation="relu")(x)
x = layers.Dropout(0.5)(x)
output = layers.Dense(1, activation="sigmoid")(x)


model = keras.Model(all_inputs, output)
model.compile("adam", "binary_crossentropy", metrics=["accuracy"])


# Visualize the connectivity graph
# keras.utils.plot_model(model, show_shapes=True, rankdir="LR")


model.fit(train_ds, epochs=50, validation_data=val_ds)
# model.save(this_folder)

attr = ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
predicted = []

with open(this_folder+"/assignment-test.csv") as testfile:
    for i, row in enumerate(testfile):
        if i==0:
            predicted.append(["RowNumber","Exited"])
            continue
        predicted.append([])
        row=row.split(",")
        row[-1]=row[-1].replace("\n","")
        predicted[-1].append(row[0])
        useful_data=row[-10:]
        sample = {}
        for j,value in enumerate(useful_data):
            if attr[j] in ['CreditScore', 'Age', 'Tenure', 'NumOfProducts', 'HasCrCard', 'IsActiveMember']:sample[attr[j]]=int(value)
            elif attr[j] in ['Balance','EstimatedSalary']: sample[attr[j]]=float(value)
            else: sample[attr[j]]=value
        input_dict = {name: tf.convert_to_tensor([value]) for name, value in sample.items()}
        predictions = model.predict(input_dict)
        predicted[-1].append(predictions[0][0])

with open(this_folder+"/submission.csv","w",newline="\n") as f:
        writer = csv.writer(f)
        writer.writerows(predicted)