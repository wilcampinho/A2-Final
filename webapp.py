# Description: This program predicts the annual amount in a Ecommerce website or app.

# Importing the Libraries.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import streamlit as st
from PIL import  Image
import pickle
import seaborn as sns

st.markdown(
        f"""
<style>
    .reportview-container .main .block-container{{
        max-width: 750px;
        padding-top: 0.5rem;
        padding-right: 0rem;
        padding-left: 0rem;
        padding-bottom: 0.5rem;
    }}
   
</style>
""",
        unsafe_allow_html=True,
    )


# Creating the Title and the Sub-Title
st.title('Money Spent Prediction on E-commerce Plateform')
st.markdown(' ### :point_left: Check the sidebar for more details!')
st.markdown(' #### Applying Regression using Python with streamlit!')
st.markdown('Adjust the parameters and the total expenditure will be diaplayed below under Prediction Amount Spent By Customer as Per Input.')



# Getting the Data
df = pd.read_csv("Ecommerce Customers")
df = df.iloc[0: , 3:]



# Get the feature input from Users
def get_user_input():
    Avg_session_length = st.slider('Avg. Session Length (In Minutes)', 25,40,33)
    Time_on_App = st.slider('Time on App (In Minutes)', 8,16,9)
    Time_on_website = st.slider('Time on Website (In Minutes)', 32,40,35)
    Membership_length = st.slider('Length of Membership (In Years)', 0.0,7.0,2.0)


    # Store a dictionary into a variable
    user_data = {'Avg_session_length':Avg_session_length,
                'Time_on_App':Time_on_App,
                'Time_on_website':Time_on_website,
                'Membership_length':Membership_length
                }

    # Transforming data into dataframe
    features = pd.DataFrame(user_data, index = [0])
    return features

# Store the user input into a variable
user_input = get_user_input()


#Setting a subheader and displaying the user input
st.subheader('Your Input:')
st.write(user_input)

# Separating X and y
X = df.iloc[:, 0:4].values
Y = df.iloc[: , -1:].values

x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.3,random_state=101)

load_clf = pickle.load(open('EC_clf.pkl', 'rb'))


#Store models predictions in a variable
predictions_of_c = load_clf.predict(user_input)

#Set a subheader ad display result
st.subheader(':point_right: Prediction Yearly Amount Spent By Customer as Per Input:' )
st.write(predictions_of_c)

# Show model metrix
st.subheader('Model Test Accuracy:')
st.write(metrics.explained_variance_score(y_test,load_clf.predict(x_test)))


#Displaying data in table
st.subheader('Data At a Glance')
st.dataframe(df.head())




# Sidebar Creation
# Side bar portion of code
author_pic = Image.open('kashyap.jpg')
st.sidebar.image(author_pic, "Your humble app creator", use_column_width=True)
st.sidebar.markdown("[Source Code](https://github.com/kashyapbd7)")
st.sidebar.write("This app uses Machine Learning!")
st.sidebar.write("It uses a Multiple regression model with Yearly amount Spent as Dependent variable.  \n\nTrained with dummy e-commerce dataset. This model was correct 98% of the \
	time when it came to predicting how much money will customer will spent per year on the plateform")
st.sidebar.markdown("This sidebar concept is taken from  \nS-DeFerrari's app  \n\n")
st.sidebar.markdown(" :point_down: Scroll Down for Visualizations")



# Displaying the chart
st.sidebar.subheader("Correlation HeatMap")
agree= st.sidebar.button('Corr Plot' )
if agree:
	st.markdown('### Correlation Plot of all the Numerical Vairables:')
	chart = plt.figure()
	sns.heatmap(df.corr(),annot=True)
	st.pyplot(chart)


numeric_columns = df.select_dtypes(['float64', 'float32', 'int32', 'int64']).columns
st.sidebar.subheader("Histogram setup")
select_box3 = st.sidebar.selectbox(label='Select Variable', options=numeric_columns)
agree2 = st.sidebar.button('Plot')
if agree2:
		st.markdown(f'### Histogram of {select_box3}')
		fig, ax = plt.subplots() #solved by add this line 
		ax=sns.histplot(data=df,x= select_box3 )
		st.pyplot(fig)


