import streamlit as st
import pickle
import numpy as np

# Load the trained model
model_path = 'model_pickle_new.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Define the main function
def main():
    st.title("ML Score Prediction")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit Bank Authenticator ML App </h2>
    </div>
    """
    
    # Input fields
    Libraries = st.number_input("Libraries", min_value=0)
    Statistics = st.number_input("Statistics", min_value=0)
    Basic_maths = st.number_input("Basic maths", min_value=0)
    supervised_algorithms = st.number_input("Supervised algorithms", min_value=0)
    unsupervised_algorithms = st.number_input("Unsupervised algorithms", min_value=0.0, format="%.2f")
    semi_supervised_algorithms = st.number_input("Semi-supervised algorithms", min_value=0)
    reinforced_algorithm = st.number_input("Reinforced algorithm", min_value=0)
    
    # When the predict button is clicked
    if st.button("Predict"):
        # Collect input features into an array
        int_features = [Libraries, Statistics, Basic_maths, supervised_algorithms, 
                        unsupervised_algorithms, semi_supervised_algorithms, reinforced_algorithm]
        final_features = [np.array(int_features)]
        
        # Make prediction
        prediction = model.predict(final_features)
        output = prediction[0]
        
        # Display the prediction
        st.success(f'Prediction: {output}')

if __name__ == "__main__":
    main()
