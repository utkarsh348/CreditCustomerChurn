import matplotlib.pyplot as plt 
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
import pandas as pd 
import numpy as np
import streamlit as st 
from PIL import Image 
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import classification_report
data = pd.read_csv("BankChurners.csv")
df = data.iloc[:,:-2]
for i in ['Gender', 'Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category']:
          df[i] = df[i].astype('category')
oe = OrdinalEncoder()
cat = df[['Gender', 'Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category']]
cat = oe.fit_transform(cat)
cat = pd.DataFrame(cat, columns = ['Gender', 'Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category'])
cols = [i for i in list(df.columns)]
for i in list(cat.columns):
  df.drop(i, axis = 1, inplace = True)
df = df.join(cat, how = 'outer')

y = df['Attrition_Flag']
X = df.iloc[:,2:]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40, stratify = y)
model = AdaBoostClassifier(n_estimators=100, random_state=1)
model.fit(X_train, y_train)

st.markdown("# Credit card customer churn")
st.markdown("## Which customers are more likely to leave? ")
st.markdown("The objective of this study is to find out what are the certain indicators that are observable for a customer that might unsubscribe from a bank's credit card service. As a bank, we wish to **retain our customers** and keep them happy, as our profits depend on their usage. Thus it becomes important for us to understand which customers are **more likely to leave** and when is the most **opportune time** to do something about it. ")
st.markdown("The study is done using a dataset created by the bank containing **user data** and **user behavior information**, containing personal details like marital status, age, gender, number of dependents, educational background and user behavior like the number of months inactive on the card, total amount of all transactions,total number of transactions, revolving balance, amount change over quarters etc.")
st.markdown("#### Attrited vs Existing Customers")
st.markdown("The following is the number of all customer that are attrited(have left the bank) or have remained a customer. ")

img1 = Image.open('./1.png')
st.image(img1)
st.markdown("As we can see there are a lot more customers who have stayed with the bank but the leaving bunch isn't by any means a small subsection. ")
st.markdown("#### Number of transactions vs frequency of customers")
st.markdown("The following is the number of transactions that have taken place by customers and how many customers have had that many transactions under their name. ")
img2 = Image.open('./2.png')
st.image(img2)
st.markdown("As we can see, the customers who chose to stay with the bank have had a much larger number of transactions, almost double that of customers that have left. This can be an indicator of the fact that such customers are either unhappy with the service or don't find themselves incentivized to use their credit card more.")
st.markdown("#### Total amount of the transactions")
st.markdown("We see similar behavior yet again as for existing customers, while the average amount of all transactions isn't all that more than attrited customers, there are quite a few customer that have had transactions of much higher value than that of attrited customers.")
img3 = Image.open('./3.png')
st.image(img3)
st.markdown("#### Total amount change from Q4 to Q1")
st.markdown("As we can see from the following graph, most attrited customers had very little change in the amount of transactions from Q4 to Q1. While a lot of existing customers behaving the same way, quite a few of them which were retained did alter the amount of transactions under their name. ")
img4 = Image.open('./4.png')
st.image(img4)
st.markdown("While it is normal to expect that certain users that spend a certain amount of money will continue to spend just as much, it is interesting to see that almost all of the users that spend above 11-12 thousand not only have very little change in the amount of transactions but also are entirely existing customers. ")
img5 = Image.open('./5.png')
st.image(img5)
st.markdown("Existing customers also have a notably higher average utilization ratio of their credit cards")

img6 = Image.open('./6.png')
st.image(img6)
st.markdown("### Personal attributes")

st.markdown("#### Customer Age")
img7 = Image.open('./7.png')
st.image(img7)
st.markdown("#### Card Category(Attrited)")
img8 = Image.open('./download (1).png')
st.image(img8)
st.markdown("#### Card Category(Existing)")
img9 = Image.open('./download (2).png')
st.image(img9)
st.markdown("#### Educational Background(Attrited)")
img9 = Image.open('./educate1.png')
st.image(img9)
st.markdown("#### Educational Background(Existing)")
img9 = Image.open('./educate2.png')
st.image(img9)
st.markdown("#### Marital status(Attrited)")
img9 = Image.open('./download (14).png')
st.image(img9)
st.markdown("#### Marital Status(Existing)")
img9 = Image.open('./download (13).png')
st.image(img9)
st.markdown("As we can see, for personal parameters, the distribution across attrited and existing customers remains more or less the same. This leads us to the conclusion that the attributes to really focus on are user behavior related.")
st.markdown("## Conclusion")
st.markdown("Since we are focusing on user behavior, The most important parameters discovered so far are transaction amounts and frequencies, utilization ratios and amount change over quarters, this points to the fact that falling usage and less incentives are the leading indicators of attrited customers. Thus, we must have systems in place that increase granularity of data collection and let us and managers know immediately when the usage is falling or there is a period of inactivity. ")
st.sidebar.markdown("## Side Panel")

st.sidebar.subheader("Pedict customer attrition")

test = []
cat = []
if st.sidebar.checkbox("Predict"):
    x1 = st.slider(label = 'Age', min_value=18, max_value= 75)
    
    x2 = st.selectbox(label = 'Gender', options = ['M','F'])
    cat.append(x2)
    x3 = st.slider(label = 'Dependents', min_value=0, max_value= 6)
    
    x4 = st.selectbox(label = 'Education', options = ['College','Doctorate','Graduate','High School','Post-Graduate','Uneducated',])
    cat.append(x4)
    x5 = st.selectbox(label = 'Marital Status', options = ['Married', 'Single', 'Unknown', 'Divorced'])
    cat.append(x5)
    x6 = st.selectbox(label = 'Income', options = ['$120K +', '$40K - $60K', '$60K - $80K', '$80K - $120K', 'Less than $40K', 'Unknown'])
    cat.append(x6)
    x7 = st.selectbox(label = 'Card', options = ['Blue', 'Gold', 'Platinum', 'Silver'])
    cat.append(x7)
    x8 = st.slider(label = 'Months on Book', min_value=1, max_value= 60)
    
    x9 = st.slider(label = 'No. of products used', min_value=1, max_value= 6)
    
    x10 = st.slider(label = 'Months inactive in the last year', min_value=1, max_value= 12)
    
    x11 = st.slider(label = 'Contacts in the last year', min_value=1, max_value= 6)

    x12 = st.slider(label = 'Credit limit', min_value=1400.0, max_value= 34600.0)
    x13 = st.slider(label = 'Revolving Balance', min_value=0.0, max_value= 3000.0)
    x14 = st.slider(label = 'Avg credit line open to buy', min_value=3.0, max_value= 34600.0)
    x15 = st.slider(label = 'Amount of change in transaction from Q4 to Q1', min_value=0.0, max_value= 4.0)
    x16 = st.slider(label = 'Total transaction amount', min_value=0.0, max_value= 20000.0)
    x17 = st.slider(label = 'Total number of transactions', min_value=0, max_value= 200)
    x18 = st.slider(label = 'Total count change from Q4 to Q1', min_value=0.0, max_value= 4.0)
    x19 = st.slider(label = 'Average utilization ratio', min_value=0.0, max_value= 1.0)
    print(cat) 
    cat = oe.transform([cat])[0]
    print(cat)
    test = [[x1,  x3, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19]]
    for i in cat:
        test[0].append(i)
    print(test)
    st.write(test)

    if st.button("Predict"):
       st.write(model.predict(test))

st.sidebar.markdown("[Data Source](https://www.kaggle.com/sakshigoyal7/credit-card-customers)")
st.sidebar.info("Author - Utkarsh Gupta | PES1UG19CS549  ")
st.sidebar.info("Made for the course - Telling Stories With Data at PES University. Fall Semester, 2021")