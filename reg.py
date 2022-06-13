import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression
import seaborn as sns
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


#im = Image.open('C:/Users/Mcastiblanco/Documents/AGPC/DataScience2020/Streamlit/Arroz/apps/arroz.png')
im2 = Image.open('r1.jpg')
st.set_page_config(page_title='Clas-App', layout="wide", page_icon=im2)
st.set_option('deprecation.showPyplotGlobalUse', False)

row1_1, row1_2 = st.columns((2, 3))

with row1_1:
    image = Image.open('r1.jpg')
    st.image(image, use_column_width=True)
    st.markdown('Web App by [Manuel Castiblanco](https://github.com/mcastiblanco1251)')
with row1_2:
    st.write("""
    # Regresión App
    Esta App utiliza algoritmos de Machine Learning  para Predecir el precio de un vehículo !
    """)
    with st.expander("Contact us 👉"):
        with st.form(key='contact', clear_on_submit=True):
            name = st.text_input('Name')
            mail = st.text_input('Email')
            q = st.text_area("Query")

            submit_button = st.form_submit_button(label='Send')
            if submit_button:
                subject = 'Consulta'
                to = 'macs1251@hotmail.com'
                sender = 'macs1251@hotmail.com'
                smtpserver = smtplib.SMTP("smtp-mail.outlook.com", 587)
                user = 'macs1251@hotmail.com'
                password = '1251macs'
                smtpserver.ehlo()
                smtpserver.starttls()
                smtpserver.ehlo()
                smtpserver.login(user, password)
                header = 'To:' + to + '\n' + 'From: ' + sender + '\n' + 'Subject:' + subject + '\n'
                message = header + '\n' + name + '\n' + mail + '\n' + q
                smtpserver.sendmail(sender, to, message)
                smtpserver.close()

st.header('Aplicación')
st.markdown('____________________________________________________________________')
app_des = st.expander('Descripción App')
with app_des:
    st.write("""Esta aplicación muestra el precio de un vehículo de acuerdo a los parámetros que se encuentran en la parte izquierda,
    el dataset es tomado de de la [UCI](https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data) para este análisis, que comprende
    las diferentes características de los vehículos y marcas. Si de das check en análisis obtendras la correlación de las diferentes variables y su análisis
    estadístico básico.
    """)

st.sidebar.header('Parámetros de Entrada Usario')

# st.sidebar.markdown("""
# [Example CSV input file](penguins_example.csv)
# """)

# Collects user input features into dataframe
uploaded_file = st.sidebar.file_uploader("Cargue sus parámetros desde un archivo CSV", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    df = pd.read_csv('car_clean.csv')
    def user_input_features():
        #island = st.sidebar.selectbox('Isla',('Biscoe','Dream','Torgersen'))
        #sex = st.sidebar.selectbox('Sexo',('Macho','Hembra'))
        symboling = st.sidebar.slider('Símbolo', int(df.symboling.min()), int(df.symboling .max()), int(df.symboling.mean()))
        wheel_base = st.sidebar.slider('Rueda Base',float(df.wheel_base.min()), float(df.wheel_base.max()), float(df.wheel_base.mean()))
        length = st.sidebar.slider('Longitud', float(df.length.min()), float(df.length.max()), float(df.length.mean()))
        width = st.sidebar.slider('Ancho', float(df.width.min()), float(df.width.max()), float(df.width.mean()))
        height = st.sidebar.slider('Altura', float(df.height.min()), float(df.height.max()), float(df.height.mean()))
        curb_weight=st.sidebar.slider('Peso', int(df.curb_weight.min()), int(df.curb_weight.max()), int(df.curb_weight.mean()))
        #engine_type=st.sidebar.slider('Tipo de Motor', float(df.engine_type.min()), float(df.engine_type.max()), float(df.engine_type.mean()))
        engine_size=st.sidebar.slider('Cilidrada', int(df.engine_size.min()), int(df.engine_size.max()), int(df.engine_size.mean()))
        #fuel_system=st.sidebar.slider('Sistema de combustible', float(df.fuel_system.min()), float(df.fuel_system.max()), float(df.fuel_system.mean()))
        bore=st.sidebar.slider('Pared cilindro Motor', float(df.bore.min()), float(df.bore.max()), float(df.bore.mean()))
        stroke=st.sidebar.slider('Tiempos', float(df.stroke.min()), float(df.stroke.max()), float(df.stroke.mean()))
        compression_ratio=st.sidebar.slider('Razón de Compresión', float(df.compression_ratio.min()), float(df.compression_ratio.max()), float(df.compression_ratio.mean()))
        horsepower=st.sidebar.slider('Potencia', int(df.horsepower.min()), int(df.horsepower.max()), int(df.horsepower.mean()))
        peak_rpm=st.sidebar.slider('Max. RPM', int(df.peak_rpm.min()), int(df.peak_rpm.max()), int(df.peak_rpm.mean()))
        city_mpg=st.sidebar.slider('Consumo Ciudad mpg', int(df.city_mpg.min()), int(df.city_mpg.max()), int(df.city_mpg.mean()))
        highway_mpg=st.sidebar.slider('Consumo Carretera mpg', int(df.highway_mpg.min()), int(df.highway_mpg.max()), int(df.highway_mpg.mean()))
        cylinder=st.sidebar.slider('Número de Cilindros', int(df.cylinder.min()), int(df.cylinder.max()), int(df.cylinder.mean()))

        data = {'symboling': symboling,
                'wheel_base': wheel_base,
                'length':length,
                'width':width,
                'height':height,
                'curb_weight':curb_weight,
                #'Tipo de Motor':engine_type,
                'engine_size':engine_size,
                #'Sistema de combustible':fuel_system,
                'bore':bore,
                'stroke':stroke,
                'compression_ratio':compression_ratio,
                'horsepower':horsepower,
                'peak_rpm':peak_rpm,
                'city_mpg':city_mpg,
                'highway_mpg':highway_mpg,
                'cylinder':cylinder,
                }
        features = pd.DataFrame(data, index=[0])
        return features
    input_df = user_input_features()

# Combines user input features with entire penguins dataset
# This will be useful for the encoding phase

car = df.drop(columns=['price'], axis=1)
df1 = pd.concat([input_df,car],axis=0)
#df[0:1]
# Encoding of ordinal features
# https://www.kaggle.com/pratik1120/penguin-dataset-eda-classification-and-clustering
# encode = ['sexo','Isla']
# for col in encode:
#     dummy = pd.get_dummies(df[col], prefix=col)
#     df = pd.concat([df,dummy], axis=1)
#     del df[col]
# df = df[:1] # Selects only the first row (the user input data)

# Displays the user input features
st.subheader('Parámetros de Entrada')

if uploaded_file is not None:
    st.write(df1)
else:
    st.write('A la espera de que se cargue el archivo CSV. Actualmente usando parámetros de entrada de ejemplo (que se muestran a continuación).')
    st.write(df1[:1])

# Reads in saved classification model
st.subheader('Análisis de Variables')
if st.checkbox("Análisis",value=False):
    st.subheader('Análisis de Correlacción')
    n=st.number_input('Parámetros a Analizar',min_value=1, max_value=16, value=int(8))
    car_df_attr= df1.iloc[:,1:n]
    car_df_attr = car_df_attr.reset_index()
    fig=sns.pairplot(car_df_attr, diag_kind = 'kde')
    st.pyplot(fig)
    st.subheader('Análisis Estadístico Descriptivo')
    st.write(df1.describe())

st.subheader('Predicción de Precio')
df2=df.iloc[:,0:16]
X = df2.drop('price', axis=1)
y = df2[['price']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)

b=np.array(df1[:1])
regression_model = LinearRegression()
a=regression_model.fit(X_train, y_train)
prediction = a.predict(b)

row1_3, row1_4 = st.columns((2, 2))

with row1_3:
    image = Image.open('r2.jpg')
    st.image(image, use_column_width=True)
with row1_4:
    st.subheader(f'El precio del carro es USD$ {prediction}')

st.subheader('Quieres conocer mas de IA, ML o DL 👉[contactanos!!](http://ia.smartecorganic.com.co/index.php/contact/)')
# for idx, col_name in enumerate(X_train.columns):
#     st.write("The coefficient for {} is {}".format(col_name, regression_model.coef_[0][idx]))
# intercept = regression_model.intercept_[0]
# st.write("The intercept for our model is {}".format(intercept))
    # r2=regression_model.score(X_test, y_test)
    #
    # st.write(f'Exactitud del modelo es {r2}')

# st.subheader('Quieres conocer mas de IA, ML o DL 👉[contactanos!!](http://ia.smartecorganic.com.co/index.php/contact/)')
