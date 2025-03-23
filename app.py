import google.generativeai as genai
import os
from flask import Flask, render_template, request
import numpy as np
from tensorflow import keras
from werkzeug.utils import secure_filename

import cv2

def analyze_apple_image(image_path):
    # Load the image
    image = cv2.imread(image_path)
    # Convert to RGB (OpenCV loads images in BGR format)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Define color ranges for red and green
    lower_red = np.array([0, 50, 50])
    upper_red = np.array([10, 255, 255])
    lower_green = np.array([35, 50, 50])
    upper_green = np.array([85, 255, 255])

    # Convert image to HSV for better color segmentation
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # Create masks for red and green colors
    red_mask = cv2.inRange(hsv_image, lower_red, upper_red)
    green_mask = cv2.inRange(hsv_image, lower_green, upper_green)

    # Calculate the percentage of red and green pixels
    total_pixels = image.shape[0] * image.shape[1]
    red_pixels = np.sum(red_mask > 0)
    green_pixels = np.sum(green_mask > 0)

    red_percentage = round((red_pixels / total_pixels) * 100, 3)
    green_percentage = round((green_pixels / total_pixels) * 100, 3)

    # Identify areas with disease (you can customize this part based on your needs)
    # Here, for example, we assume a simple threshold for disease detection
    # You might replace this with actual disease detection logic
    disease_mask = cv2.inRange(image, np.array([0, 0, 0]), np.array([50, 50, 50]))  # Adjust as necessary
    disease_area = np.sum(disease_mask > 0)
    disease_size = round((disease_area / total_pixels) * 100, 3)  # Percentage of the area affected

    # Return the analysis results
    return {
        "red_percentage": red_percentage,
        "green_percentage": green_percentage,
        "disease_size": disease_size,
        "total_pixels": total_pixels
    }




def calculate_nutritional_impact(analysis_results):
    # Example logic to determine nutritional impact based on analysis results
    red_percentage = analysis_results["red_percentage"]
    green_percentage = analysis_results["green_percentage"]
    disease_size = analysis_results["disease_size"]

    # Assume a base nutritional value for a healthy apple
    base_nutrition = {
        "Calories": 95,
        "Carbohydrates": "25g",
        "Fiber": "4g",
        "Vitamins": "A, C, K",
        "Minerals": "Potassium, Calcium"
    }

    # Adjust nutritional values based on color and disease size
    if disease_size > 10:  # Example threshold for disease impact
        base_nutrition["Calories"] -= 20  # Decrease calories if disease is significant
        base_nutrition["Fiber"] = str(float(base_nutrition["Fiber"].replace("g", "")) - 1) + "g"  # Decrease fiber
        base_nutrition["Vitamins"] = "Lower levels"  # General note

    # Further adjustments can be made based on red and green percentages

    return base_nutrition

# Configure the Google Generative AI with API key
os.environ["API_KEY"] = 'AIzaSyBeat7bxnm7CCBe6MXzFrZO4N_gYkcze1I'  # Replace with your actual API key
genai.configure(api_key=os.environ["API_KEY"])

# Initialize the Google Generative AI model (renamed to avoid conflicts)
genai_model = genai.GenerativeModel('gemini-1.5-flash-latest')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Load the pre-trained apple disease classification model
model_path = 'C:/Users/mohdd/PycharmProjects/INT/apple_latest.h5'  # Update path if needed
model = keras.models.load_model(model_path)

# Define disease classes
disease_classes = ["BLOTCH", "NORMAL", "ROT", "SCAB"]


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


@app.route('/detect', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image' not in request.files:
            return "No file part"
        file = request.files['image']
        if file.filename == '':
            return "No selected file"
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            # Perform disease detection (existing function)
            predicted_class = detect_disease(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            print(f"Predicted Disease Class: {predicted_class}")

            # Analyze the uploaded apple image
            analysis_results = analyze_apple_image(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            # Calculate nutritional impact based on the analysis
            nutrition_info = calculate_nutritional_impact(analysis_results)

            return render_template('index.html',
                                   prediction=predicted_class,
                                   additional_info=get_disease_info(predicted_class),
                                   nutrition_info=nutrition_info,
                                   analysis_results=analysis_results)
    return render_template('index.html', prediction=None, additional_info=None, nutrition_info=None,
                           analysis_results=None)


def detect_disease(image_path):
    # Load and preprocess the image
    image = keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    image = keras.preprocessing.image.img_to_array(image)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)

    # Perform disease classification
    prediction = model.predict(image)
    predicted_class_index = np.argmax(prediction)
    predicted_class = disease_classes[predicted_class_index]
    print(f"Predicted Disease Class: {predicted_class}")  # Already added in detect_disease()

    return predicted_class  # Ensure this returns the predicted disease


def get_disease_info(disease):
    prompt = f"""
    Start the response with the predicted disease name.like: Predicted disease is apple {disease} .then For the apple disease {disease}, please provide the following information in bullet points for a HTML page:
    1. Remedies: What are the effective treatments or solutions for this disease?
    2. Effects: How does this disease affect the apple tree, leaves, or fruit?
    3. Best Practices: What are the recommended steps to prevent or manage this disease in an orchard or garden?
    """

    try:
        # Generate textual content for remedies, effects, and best practices
        response = genai_model.generate_content(prompt)
        result = response.text.strip()

        # Initialize formatted response
        remedies = effects = best_practices = ""

        # Properly format the remedies
        if "Remedies:" in result:
            remedies_part = result.split("Remedies:")[1].split("Effects:")[0].strip()
            remedies_list = remedies_part.split('\n')  # Split into individual points
            remedies_items = "".join([f"<li>{item.replace('*', '').strip()}</li>" for item in remedies_list])
            remedies = f"<h3>Remedies:</h3><ul>{remedies_items}</ul>"

        # Properly format the effects
        if "Effects:" in result:
            effects_part = result.split("Effects:")[1].split("Best Practices:")[0].strip()
            effects_list = effects_part.split('\n')  # Split into individual points
            effects_items = "".join([f"<li>{item.replace('*', '').strip()}</li>" for item in effects_list])
            effects = f"<h3>Effects:</h3><ul>{effects_items}</ul>"

        # Properly format the best practices
        if "Best Practices:" in result:
            best_practices_part = result.split("Best Practices:")[1].strip()
            best_practices_list = best_practices_part.split('\n')  # Split into individual points
            best_practices_items = "".join(
                [f"<li>{item.replace('*', '').strip()}</li>" for item in best_practices_list])
            best_practices = f"<h3>Best Practices:</h3><ul>{best_practices_items}</ul>"

        formatted_response = f"{remedies}{effects}{best_practices}"

        return formatted_response, None  # Return None for image URL

    except Exception as e:
        # Return an error message and None for the image URL
        return f"Error fetching information: {str(e)}", None


if __name__ == '__main__':
    print("Starting Flask app...")
    print("Click here: http://127.0.0.1:5000/detect")
    app.run(host='127.0.0.1', port=5000, debug=True)
