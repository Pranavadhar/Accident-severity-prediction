import pandas as pd
import gradio as gr
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

def accident_severity_predictor(sex, num_vehicles_involved, num_casualties):
    df = pd.read_csv('RTA Dataset modified.csv')
    X = df[['Sex_of_driver', 'Number_of_vehicles_involved', 'Number_of_casualties']]
    encoder = OneHotEncoder(sparse=False, drop='first')
    X_encoded = encoder.fit_transform(X[['Sex_of_driver']])
    y = df['Accident_severity']
    X_train, _, y_train, _ = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    user_input_df = pd.DataFrame({'Sex_of_driver': [sex], 'Number_of_vehicles_involved': [num_vehicles_involved], 'Number_of_casualties': [num_casualties]})
    user_input_encoded = encoder.transform(user_input_df[['Sex_of_driver']])
    prediction = model.predict(user_input_encoded)
    return prediction[0]

sex_input = gr.Textbox(label="Enter the Sex (Male, Female, Unknown)")
num_vehicles_input = gr.Number(label="Enter the number of vehicles involved")
num_casualties_input = gr.Number(label="Enter the number of casualties involved")
output = gr.Textbox(label="Predicted Severity")

iface = gr.Interface(fn=accident_severity_predictor,
                     inputs=[sex_input, num_vehicles_input, num_casualties_input],
                     outputs=output,
                     title="ROAD TRAFFIC ACCIDENT SEVERITY PREDICTION",
                     description="This is an END to END Machine Learning project done to predict the accident severity of a person by giving the inputs in the prompt. \n\n"
                                 "This Accident severity project has been deployed to showcase the main objective of the severity happened to the person who met with an accident. \n\n"
                                 "MODEL DEVELOPMENT AND MODEL DEPLOYMENT: PRANAVADHAR A. \n\n"
                                 "DEPLOYMENT TOOL: GRADIO \n\n"
                                 "GITHUB: https://github.com/Pranavadhar \n\n"
                                 "LINKED IN: https://www.linkedin.com/in/pranavadhar-a-b19525289/ \n\n")

iface.launch(share=True)
