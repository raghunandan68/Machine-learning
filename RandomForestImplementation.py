import streamlit as st
from sklearn.model_selection import train_test_split,RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,r2_score,mean_squared_error
st.set_page_config(page_title="Random Forest",layout="centered")
import pandas as pd
df=pd.read_csv("housing.csv")
st.title("Random Forest Classification and Regression")
st.write(df.head())
st.sidebar.header("Model Settings")
model=st.sidebar.selectbox("Select Model",["RandomForestClassifier","RandomForestRegressor"])
size=st.sidebar.slider("Test size",0.0,0.5,0.3)
r=st.sidebar.checkbox("Use Randomized Search CV")
if model=="RandomForestClassifier":
    if st.button("Train Model"):
        if not r:
            x=df.drop(['ocean_proximity'],axis=1)
            y=df['ocean_proximity']
            x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=size,random_state=42)
            r1=RandomForestClassifier()
            r1.fit(x_train,y_train)
            y_pred=r1.predict(x_test)
            st.subheader("Accuracy Score")
            st.write(accuracy_score(y_test,y_pred))
            st.subheader("Classificaton Report")
            st.write(classification_report(y_test,y_pred))
            st.subheader("Confusion Matrix")
            st.write(confusion_matrix(y_test,y_pred))
        else:
            x=df.drop(['ocean_proximity'],axis=1)
            y=df['ocean_proximity']
            x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=size,random_state=42)
            params={
                'criterion':['gini','entropy','log_loss'],
                'max_depth':[10,20,30],
                'max_features':['auto','sqrt','log2']
            }
            r2=RandomForestClassifier()
            ra=RandomizedSearchCV(r2,param_distributions=params,cv=3)
            ra.fit(x_train,y_train)
            y_pred=ra.predict(x_test)
            st.subheader("Accuracy Score")
            st.write(accuracy_score(y_test,y_pred))
            st.subheader("Classificaton Report")
            st.write(classification_report(y_test,y_pred))
            st.subheader("Confusion Matrix")
            st.write(confusion_matrix(y_test,y_pred))
else:
    if st.button("Train Model"):
        if not r:
            c=['median_house_value','ocean_proximity']
            x=df.drop(columns=c,axis=1)
            y=df['median_house_value']
            x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=size,random_state=42)
            r1=RandomForestRegressor()
            r1.fit(x_train,y_train)
            y_pred=r1.predict(x_test)
            st.subheader("R2 Score")
            st.write(r2_score(y_test,y_pred))
            st.subheader("Mean Squared Error")
            st.write(mean_squared_error(y_test,y_pred))
        else:
            c=['median_house_value','ocean_proximity']
            x=df.drop(columns=c,axis=1)
            y=df['median_house_value']
            x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=size,random_state=42)
            params={
                'criterion':['squared_error'],
                'min_samples_split':[2,5],
                'min_samples_leaf':[1,2],
                'n_estimators':[50]
            }
            r2=RandomForestRegressor()
            ra=RandomizedSearchCV(r2,param_distributions=params,cv=3)
            ra.fit(x_train,y_train)
            y_pred=ra.predict(x_test)
            st.subheader("R2 Score")
            st.write(r2_score(y_test,y_pred))
            st.subheader("Mean Squared Error")
            st.write(mean_squared_error(y_test,y_pred))