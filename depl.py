import gradio as gr
import pickle


def make_prediction(Sex_of_driver, Number_of_vehicles_involved, Number_of_casualties):
    with open("rars.pkl", "rb") as f:
        clf = pickle.load(f)
        preds = clf.predict(
            [[Sex_of_driver, Number_of_vehicles_involved, Number_of_casualties]])
        print(preds)
        return preds

# Create the input component for Gradio since we are expecting 4 inputs


SEX_input = gr.Number(
    label="Enter the SEX {1:For Male, 2: For Female, 3: For Unknown}")
NO_OF_VEHICLE_INVOLVED_input = gr.Number(
    label="Enter the number of vehicles involved")
NO_OF_CASUALITY_input = gr.Textbox(
    label="Enter the number of casualties involved")

# We create the output
output = gr.Textbox()

app = gr.Interface(fn=make_prediction, inputs=[
                   SEX_input, NO_OF_VEHICLE_INVOLVED_input, NO_OF_CASUALITY_input], outputs=output, title="ROAD TRAFIC ACCIDENT SEVERITY PREDICTION",
                   description="This is an END to END Machine Learning project done to predict the accident severity of a person by giving the inputs in the prompt. \n\n"
                   "This Accident severity project has been deployed to showcase the main objective of the severity happened to the person who met with an accident. \n\n"
                   "MODEL DEVELOPMENT AND MODEL DEPLOYMENT: PRANAVADHAR A. \n\n"
                   "DEPLOYMENT TOOL: GRADIO \n\n"
                   "GITHUB: https://github.com/Pranavadhar \n\n"
                   "LINKED IN: https://www.linkedin.com/in/pranavadhar-a-b19525289/ \n\n")
app.launch(share=True)
