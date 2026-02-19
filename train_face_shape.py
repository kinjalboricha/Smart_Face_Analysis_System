import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import joblib

data = {
    "ratio":[1.5,1.4,1.3,1.1,1.0,1.2],
    "shape":[0,0,0,1,2,1]
}

df=pd.DataFrame(data)

X=df[["ratio"]]
y=df["shape"]

model=KNeighborsClassifier(n_neighbors=3)
model.fit(X,y)

joblib.dump(model,"face_shape_model.pkl")

print("Face Shape Model Saved")