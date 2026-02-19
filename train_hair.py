import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import joblib

data={
    "shape":[0,1,2],
    "hair":[0,1,2]
}

df=pd.DataFrame(data)

X=df[["shape"]]
y=df["hair"]

model=DecisionTreeClassifier()
model.fit(X,y)

joblib.dump(model,"hair_model.pkl")

print("Hair Model Saved")