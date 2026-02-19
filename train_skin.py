print("File is running....")
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import joblib

data={
    "light":[180,160,140,120,100],
    "tone":[0,0,1,1,2]
}

df=pd.DataFrame(data)

X=df[["light"]]
y=df["tone"]

model=KNeighborsClassifier(3)
model.fit(X,y)

joblib.dump(model,"skin_model.pkl")

print("Skin Model Saved")