import os  # Import the os module for interacting with the operating system
import numpy as np  # Import numpy for numerical operations
from tensorflow.keras.models import load_model  # Import load_model from keras to load the trained model
from tensorflow.keras.preprocessing import image  # Import image preprocessing utilities from keras

# Load the best model
model_dir = 'model'  # Directory where the model is stored
model = load_model(os.path.join(model_dir, 'best_model.keras'))  # Load the model from the specified path

# Class names (replace with actual class names from your dataset, including 'Healthy')
class_names = ['Anthracnose', 'Bacterial Canker', 'Cutting Weevil', 'Die Back', 'Gall Midge', 'Powdery Mildew', 'Sooty Mould', 'Healthy']  # List of disease names and 'Healthy'

# Confidence threshold
confidence_threshold = 0.5  # Define the confidence threshold for classification

# Management techniques for each disease
management_techniques = {
    'Anthracnose': "Apply fungicides such as chlorothalonil or copper-based fungicides. Ensure proper pruning and avoid overhead irrigation.",  # Management for Anthracnose
    'Bacterial Canker': "Use copper-based bactericides. Prune and destroy infected branches. Ensure good air circulation.",  # Management for Bacterial Canker
    'Cutting Weevil': "Use insecticides like carbaryl. Remove and destroy affected fruits and branches.",  # Management for Cutting Weevil
    'Die Back': "Apply fungicides such as carbendazim. Prune and burn affected branches. Ensure proper fertilization and irrigation.",  # Management for Die Back
    'Gall Midge': "Use insecticides like lambda-cyhalothrin. Remove and destroy affected plant parts. Promote good field sanitation.",  # Management for Gall Midge
    'Powdery Mildew': "Apply sulfur or potassium bicarbonate. Ensure good air circulation and avoid excessive nitrogen fertilization.",  # Management for Powdery Mildew
    'Sooty Mould': "Control the insect pests that produce honeydew, such as aphids and whiteflies. Wash off the mould with water.",  # Management for Sooty Mould
    'Healthy': "No action needed. The mango is healthy."  # Action for Healthy
}

# Function to classify an image and provide management techniques
def classify_image(img_path):  # Define the function to classify an image
    try:  # Try block to handle exceptions
        img = image.load_img(img_path, target_size=(150, 150))  # Load the image with target size 150x150
        img_array = image.img_to_array(img)  # Convert the image to an array
        img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize the image array and add batch dimension

        predictions = model.predict(img_array)  # Predict the class of the image using the model
        max_confidence = np.max(predictions)  # Get the maximum confidence score from the predictions
        predicted_class = np.argmax(predictions, axis=1)  # Get the index of the class with the highest score

        if max_confidence < confidence_threshold:  # Check if the confidence is below the threshold
            return "Unhealthy", "The image is classified as unhealthy. Further analysis is needed."  # Return Unhealthy if below threshold
        else:  # If confidence is above the threshold
            disease = class_names[predicted_class[0]]  # Get the class name corresponding to the highest score
            management = management_techniques.get(disease, "No management technique available.")  # Get the management technique for the disease
            return disease, management  # Return the predicted class and management technique
    except Exception as e:  # Catch any exceptions that occur
        return f"Error: {str(e)}", ""  # Return the error message

# Main code to take user input and classify the image
if __name__ == "__main__":  # Check if the script is being run directly
    img_path = input("Enter the path of the image: ")  # Prompt the user to enter the path of the image
    if os.path.exists(img_path):  # Check if the provided image path exists
        predicted_class, management = classify_image(img_path)  # Classify the image and get the management technique
        print(f'The predicted class for the image is: {predicted_class}')  # Print the predicted class
        print(f'Management technique: {management}')  # Print the management technique
    else:  # If the provided image path does not exist
        print("The provided image path does not exist. Please check and try again.")  # Inform the user that the path does not exist
