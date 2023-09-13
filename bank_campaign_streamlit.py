# Deploy Travel Insurance Purchase Prediction 

# --------------------------------------------

import numpy as np
import pandas as pd
import base64

import pickle
import streamlit as st

# --------------------------------------------

st.title('''
         BANK CAMPAIGN TERM DEPOSIT PREDICTOR
         ''')

# sidebar

st.sidebar.header("Please input customer's feature")
Age = st.sidebar.slider('Age', 17, 98, 17)
Job = st.sidebar.selectbox('Job', ['housemaid', 'services', 'admin', 'blue-collar', 'technician', 'retired','management', 'unemployed', 'self-employed', 'unknown', 'entrepreneur', 'student'])
Marital = st.sidebar.radio('Marital Status', ['married', 'single', 'divorced', 'unknown'])
Education = st.sidebar.selectbox('Education', ['illiterate', 'basic.4y',  'basic.6y', 'basic.9y', 'high.school', 'professional.course', 'university.degree', 'unknown'])
Default = st.sidebar.radio('Have Default?', ['yes','no','unknown'])
Housing = st.sidebar.radio('Have any Housing?', ['yes','no','unknown'])
Loan = st.sidebar.radio('Have any Loan?', ['yes','no','unknown'])
Contact = st.sidebar.radio('Contact method', ['telephone', 'cellular'])
Month = st.sidebar.selectbox('Month of last contact', ['mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])
Day_of_Week = st.sidebar.selectbox('Day of Week of last contact', ['mon', 'tue', 'wed', 'thu', 'fri'])
Campaign = st.sidebar.slider('How many contact done in this campaign?', 1, 56, 3)
pdays= st.sidebar.number_input('How many days since last contact? (999 if has never been contacted)', min_value=0, max_value=999)
previous = st.sidebar.slider('How many contact done in last campaign? (0 if has never been contacted)', 0, 7, 3)
poutcome = st.sidebar.radio('Is last campaign a success for this customer?', ['success', 'failure', 'nonexistent'])
emp_var_rate = st.sidebar.number_input('Employment Variation Rate at that time', min_value=-3.4, max_value=1.4)
CPI_index = st.sidebar.number_input('Consumer Price Index at that time', min_value=92.201 , max_value=94.767)
CCI_index = st.sidebar.number_input('Consumer Confidence Index at that time', min_value=-50.8 , max_value=-26.9)
euribor3m = st.sidebar.number_input('Euribor 3 month interest rate', min_value=0.634 , max_value=5.045)
nr_employed = st.sidebar.number_input('Euribor 3 month interest rate', min_value=4963.6 , max_value=5228.1)


# ==============================================

def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()


img = get_img_as_base64("sidebar.jpg")
img2 = get_img_as_base64("banking2.jpg")

page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("data:image/png;base64,{img2}");
background-size: 125%;
background-position: top left;
background-repeat: no-repeat;
background-attachment: local;
}}

[data-testid="stSidebar"] > div:first-child {{
background-image: url("data:image/png;base64,{img}");
background-position: center; 
background-repeat: no-repeat;
background-attachment: fixed;
}}

[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}

[data-testid="stToolbar"] {{
right: 2rem;
}}
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)



# membuat dan menampilkan dataframe
df = pd.DataFrame()
df['age'] = [Age]
df['job'] = [Job]
df['marital'] = [Marital]
df['education'] = [Education]
df['default'] = [Default]
df['housing'] = [Housing]
df['loan'] = [Loan]
df['contact'] = [Contact]
df['month'] = [Month]
df['day_of_week'] = [Day_of_Week]
df['campaign'] = [Campaign]
df['pdays'] = [pdays]
df['previous'] = [previous]
df['poutcome'] = [poutcome]
df['emp.var.rate'] = [emp_var_rate]
df['cons.price.idx'] = [CPI_index]
df['cons.conf.idx'] = [CCI_index]
df['euribor3m'] = [euribor3m]
df['nr.employed'] = [nr_employed]

df.index = ['Value']

st.dataframe(df.T, width=400)

# load model (masukkan path dari model di dalam r'')
with open(r'C:\Theo\Purwadhika\Purwadhika\Modul3\final_model_for_Bank_Campaign_Final_Project.sav', 'rb') as f:
    model_loaded = pickle.load(f)

# predict
kelas = model_loaded.predict(df)

# show result
st.subheader('Prediction Result :')

if kelas == 1 : 
    st.write('Class 1 : this customer is predicted to do TERM DEPOSIT')
    st.write('''Action : Hubungi Nasabah dan Usahakan untuk menghubungi pada jam yang tidak 
             sibuk serta membicarakan topik yang menarik untuk nasabah sesuai 
             dengan karakteristiknya dan usahakan durasi percakapan 8 - 10 menit.
             ''')
else :
    st.write('Class 0 : this customer is predicted NOT going to do TERM DEPOSIT')
    st.write('''Action : Tidak dianjurkan untuk menghubungi nasabah tersebut. Tetapi jika seandainya
             ada pertimbangan lain, boleh dihubungi dengan durasi percakapan tidak lebih lama dari 
             3 menit jika tidak terindikasi adanya ketertarikan.
             ''')

