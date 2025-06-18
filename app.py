# import streamlit as st
# import pandas as pd
# import numpy as np
# import pickle
# import base64

# # Function to create a download Link for a DataFrame as a CSV file
# def get_binary_file_downloader_html(df):
#     csv = df.to_csv(index=False)
#     b64 = base64.b64encode(csv.encode()).decode()
#     href = f'<a href="data:file/csv;base64, {b64}" download="predictions.csv">Download Predictions CSV</a>'
#     return href 

# st.title("Heart Disease Predictor")
# tab1,tab2,tab3 = st.tabs(['Individual Predict','Bulk Predict','model Information'])

# with tab1:
#     age = st.number_input("Age (years)", min_value=0, max_value=150)
#     sex = st.selectbox("Sex", ["Male", "Female"])
#     chest_pain = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"])
#     resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", min_value=0, max_value=300)
#     cholesterol = st.number_input("Serum Cholesterol (mm/dl)", min_value=0)
#     fasting_bs = st.selectbox("Fasting Blood Sugar", ["<= 120 mg/dl", "> 120 mg/dl"])
#     resting_ecg = st.selectbox("Resting ECG Results", ["Normal", "ST-T wave Abnormality", "Left Ventricular Hypertrophy"])
#     max_hr = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=202)
#     exercise_angina = st.selectbox("Exercise-Induced Angina", ["Yes", "No"])
#     oldpeak = st.number_input("Oldpeak (ST Depression)", min_value=0.0, max_value=10.0) 
#     st_slope = st.selectbox("Slope of Peak Exercise ST Segment", ["Upsloping", "Flat", "Downsloping"])

#     #Convert categorical inputs to numeric
#     sex = 0 if sex == "Male" else 1
#     chest_pain = ["Atypical Angina", "Non-Anginal Pain", "Asymptomatic", "Typical Angina"].index(chest_pain)
#     fasting_bs = 1 if fasting_bs == "> 120 mg/dl" else 0
#     resting_ecg = ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"].index(resting_ecg)
#     exercise_angina = 1 if exercise_angina == "Yes" else 0
#     st_slope = ["Upsloping", "Flat", "Downsloping"].index(st_slope)

#     #Create a DataFrame with user inputs
#     input_data = pd.DataFrame({
#         'age': [age],
#         'sex': [sex],
#         'chest pain': [chest_pain],
#         'resting bp': [resting_bp],
#         'cholesterol': [cholesterol],
#         'fasting bs': [fasting_bs],
#         'resting_ecg': [resting_ecg],
#         'max_hr': [max_hr],
#         'exercise_angina': [exercise_angina],
#         'oldpeak': [oldpeak],
#         'st_slope': [st_slope]
#     })

#     algonames = ['Decision Trees', 'Logistic Regression', 'Random Forest', 'Support Vector Machine', 'Grid Random Forest']
#     modelnames = ['DecisionTree.pkl', 'LogisticRegression.pkl', 'RandomForest.pkl', 'SVM.pkl', 'gridRF.pkl']

#     predictions = []
#     def predict_heart_disease(data):
#         for modelname in modelnames:
#             model = pickle.load(open(modelname, 'rb'))
#             prediction = model.predict(data)
#             predictions.append(prediction)
#         return predictions
    
#     #Create a submit button to make predictions
#     if st.button("Submit"):
#         st.subheader('Results....')
#         st.markdown('--------------------')

#         # Rename to match the model's training feature names
#         input_data.columns = ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol',
#                       'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope']

#         # Then make prediction
#         result = predict_heart_disease(input_data)

        
#         # result = predict_heart_disease(input_data)
    
#     for i in range(len(predictions)):
#         st.subheader(algonames[i])
#         if result[i][0]== 0:
#             st.write("No heart disease detected.")
#         else:
#             st.write("Heart disease detected.")
#         st.markdown('----------------------')

# with tab2:
#     st.title("Upload CSV File")
    
#     st.subheader('Instructions to note before uploading the file:')
#     st.info("""
#         1. No NaN values allowed.
#         2. Total 11 features in this order ('Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope').\n
#         3. Check the spellings of the feature names.
#         4. Feature values conventions: \n     
#             - Age: age of the patient [years] \n
#             - Sex: sex of the patient [0: Male, 1: Female] \n
#             - ChestPainType: chest pain type [3: Typical Angina, 0: Atypical Angina, 1: Non-Anginal Pain, 2: Asymptomatic]\n
#             - RestingBP: resting blood pressure [mm Hg] \n
#             - Cholesterol: serum cholesterol [mm/dl] \n
#             - FastingBS: fasting blood sugar [1: if FastingBS > 120 mg/dl, 0: otherwise] \n
#             - RestingECG: resting electrocardiogram results [0: Normal, 1: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of >0.05 mV),2:showing probable or definite left ventricular hypertrophy by Este's criteria]\n
#             - MaxHR: maximum heart rate achieved [Numeric value between 60 and 202] \n
#             - ExerciseAngina: exercise-induced angina [1: Yes, 0: No] \n
#             - Oldpeak: oldpeak = ST [Numeric value measured in depression] \n
#             - ST_Slope: the slope of the peak exercise ST segment [8: upsloping, 1: flat, 2: downsloping] \n
            
    
            
# """)
#     # create a file uploader in the sidebar
#     uploaded_file = st.file_uploader("Upload a csv file", type=["csv"])
    
#     if uploaded_file is not None:
#         #Read the uploaded CSV file into a DataFrame
#         input_data = pd.read_csv(uploaded_file) 
#         model = pickle.load(open('LogisticRegression.pkl','rb'))
        
#         #Ensure that the input DataFrame matches the expected columns and format
#         expected_columns = ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS', 
#                             'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope']
        
#         if set(expected_columns).issubset(input_data.columns):
#             input_data['Prediction LR'] = ''
            
#             for i in range(len(input_data)):
#                 arr = input_data.iloc[i,:-1].values
#                 # input_data ['Predictionl LR'][i] = model.predict([arr])[0]
#                 input_data.loc[i, 'Prediction LR'] = model.predict([arr])[0]

#             input_data.to_csv('PredictedHeartLR.csv')

#             # Display the predictions
#             st.subheader("Predictions:")
#             st.write(input_data)

#             #Create a button to download the updated CSV file
#             st.markdown(get_binary_file_downloader_html(input_data), unsafe_allow_html=True)
#         else:
#             st.warning("Please make sure the uploaded CSV file has the correct columns.")
            
#     else:
#         st.info("Upload a CSV file to get predictions.")


# with tab3:
#     import plotly.express as px
#     data = {'Decision Trees': 80.97, 'Logistic Regression': 85.86, 'Random Forest': 84.23, 'Support Vector Machine': 84.22, 'GridRF':89.57}
#     Models = list(data.keys())
#     Accuracies = list(data.values())
#     df = pd.DataFrame(list(zip(Models, Accuracies)),columns=['Models', 'Accuracies'])
#     fig = px.bar(df,y='Accuracies', x='Models')
#     st.plotly_chart(fig)









import streamlit as st
import pandas as pd
import numpy as np
import pickle
import base64
import plotly.express as px

# ------------ THEME SIMULATION (light/dark toggle) ------------
# theme = st.sidebar.radio("🌓 Choose Theme", ["Light", "Dark"])
# if theme == "Dark":
#     st.markdown("""
#         <style>
#         body, .stApp { background-color: #0e1117; color: white; }
#         </style>
#     """, unsafe_allow_html=True)

# ------------ DOWNLOAD LINK FUNCTION ------------
def get_binary_file_downloader_html(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="predictions.csv">📥 Download Predictions CSV</a>'
    return href 

st.title("Heart 💝 Disease Predictor")

# Tabs
tab1, tab2, tab3 = st.tabs(['📋 Individual Predict','📁 Bulk Predict','📊 Model Information'])

# ------------ MODEL NAME MAPPING ------------
model_options = {
    'Decision Tree( Accuracy: 80.97%)': 'DecisionTree.pkl',
    'Support Vector Machine( Accuracy: 84.22%)': 'SVM.pkl',
    'Random Forest( Accuracy: 84.23%)': 'RandomForest.pkl',
    'Logistic Regression( Accuracy: 85.86%)': 'LogisticRegression.pkl',
    'Grid Random Forest( Accuracy: 89.57%)': 'gridRF.pkl'
}

# ------------ INDIVIDUAL PREDICTION ------------
with tab1:
    st.subheader("Enter Patient Information")

   

    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age (years)", min_value=0, max_value=150)
        sex = st.selectbox("Sex", ["Male", "Female"])
        chest_pain = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"])
        resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", min_value=0, max_value=300)
        cholesterol = st.number_input("Serum Cholesterol (mg/dl)", min_value=0)
        fasting_bs = st.selectbox("Fasting Blood Sugar", ["<= 120 mg/dl", "> 120 mg/dl"])
    with col2:
        resting_ecg = st.selectbox("Resting ECG Results", ["Normal", "ST-T wave Abnormality", "Left Ventricular Hypertrophy"])
        max_hr = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=202)
        exercise_angina = st.selectbox("Exercise-Induced Angina", ["Yes", "No"])
        oldpeak = st.number_input("Oldpeak (ST Depression)", min_value=0.0, max_value=10.0)
        st_slope = st.selectbox("Slope of Peak Exercise ST Segment", ["Upsloping", "Flat", "Downsloping"])

    # Encode categorical values
    sex = 0 if sex == "Male" else 1
    chest_pain = ["Atypical Angina", "Non-Anginal Pain", "Asymptomatic", "Typical Angina"].index(chest_pain)
    fasting_bs = 1 if fasting_bs == "> 120 mg/dl" else 0
    resting_ecg = ["Normal", "ST-T wave Abnormality", "Left Ventricular Hypertrophy"].index(resting_ecg)
    exercise_angina = 1 if exercise_angina == "Yes" else 0
    st_slope = ["Upsloping", "Flat", "Downsloping"].index(st_slope)

    input_data = pd.DataFrame({
        'Age': [age],
        'Sex': [sex],
        'ChestPainType': [chest_pain],
        'RestingBP': [resting_bp],
        'Cholesterol': [cholesterol],
        'FastingBS': [fasting_bs],
        'RestingECG': [resting_ecg],
        'MaxHR': [max_hr],
        'ExerciseAngina': [exercise_angina],
        'Oldpeak': [oldpeak],
        'ST_Slope': [st_slope]
    })

    selected_model = st.selectbox("Select Model for Prediction", list(model_options.keys()))

    if st.button("🔍 Predict"):
        model_file = model_options[selected_model]
        model = pickle.load(open(model_file, 'rb'))
        result = model.predict(input_data)[0]

        st.subheader(f"Result using {selected_model}")
        if result == 0:
            st.success("No heart disease detected.")
        else:
            st.error("⚠️ Heart disease detected.")

# ------------ BULK PREDICTION (CSV or XLSX) ------------
with tab2:
    # st.title(" Upload File for Bulk Prediction")
    st.subheader("Instructions:")
    st.info("""
1. File types: `.csv` or `.xlsx`
2. No missing (NaN) values allowed.
3. Must have **11 columns** in this order:
   ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol',
   'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope']
4. Feature value formats:
     - Age: age of the patient [years] \n
     - Sex: sex of the patient [0: Male, 1: Female] \n
     - ChestPainType: chest pain type [0: Atypical Angina, 1: Non-Anginal Pain, 2: Asymptomatic, 3: Typical Angina]\n
     - RestingBP: resting blood pressure [mm Hg] \n
     - Cholesterol: serum cholesterol [mm/dl] \n
     - FastingBS: fasting blood sugar [1: if FastingBS > 120 mg/dl, 0: otherwise] \n
     - RestingECG: resting electrocardiogram results [0: Normal, 1: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of >0.05 mV),2:showing probable or definite left ventricular hypertrophy by Este's criteria]\n
     - MaxHR: maximum heart rate achieved [Numeric value between 60 and 202] \n
     - ExerciseAngina: exercise-induced angina [1: Yes, 0: No] \n
     - Oldpeak: oldpeak = ST [Numeric value measured in depression] \n
     - ST_Slope: the slope of the peak exercise ST segment [8: upsloping, 1: flat, 2: downsloping] \n
            
    """)

    uploaded_file = st.file_uploader("📤 Upload a CSV or Excel file", type=["csv", "xlsx"])
    if uploaded_file is not None:
        # Load either CSV or Excel
        if uploaded_file.name.endswith('.csv'):
            input_data = pd.read_csv(uploaded_file)
        else:
            input_data = pd.read_excel(uploaded_file)

        expected_columns = ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS', 
                            'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope']

        if set(expected_columns).issubset(input_data.columns):
            model = pickle.load(open('LogisticRegression.pkl','rb'))
            input_data['Logistic Regression Predict'] = model.predict(input_data[expected_columns])

            st.subheader("✅ Predictions Complete")
            st.write(input_data)

            st.markdown(get_binary_file_downloader_html(input_data), unsafe_allow_html=True)
        else:
            st.warning("⚠️ Uploaded file does not match expected column names.")
    else:
        st.info("Upload a valid CSV or Excel file to proceed.")

# ------------ MODEL INFO TAB ------------
with tab3:
    st.subheader("📊 Model Accuracy Comparison")
    data = {
        'Decision Tree': 80.97,
        'Logistic Regression': 85.86,
        'Random Forest': 84.23,
        'Support Vector Machine': 84.22,
        'Grid Random Forest': 89.57
    }
    df = pd.DataFrame(list(data.items()), columns=['Model', 'Accuracy'])

    fig = px.bar(df, x='Model', y='Accuracy', text='Accuracy', color='Model', color_discrete_sequence=px.colors.qualitative.Bold)
    fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
    fig.update_layout(yaxis_range=[75, 95])
    st.plotly_chart(fig, use_container_width=True)

       





