# README: Flask Chatbot for Lithium Battery Company

## Overview
This project is a chatbot application built with Flask and integrated with Hugging Face's GPT-2 model for text generation. The chatbot is designed to respond to customer inquiries for a lithium battery company, assisting users with various topics related to the products and services offered by the company. It uses a combination of NLP techniques like intent classification and text generation to generate responses to user inputs.

### Features:
- Intent classification for various customer queries (e.g., product inquiries, shipping details, pricing, etc.)
- Response generation using GPT-2 from Hugging Face
- Flask-based web application for user interaction
- Custom training on a predefined set of intents to improve accuracy

---

## Prerequisites

Before running the Flask chatbot, make sure you have the following installed:

- **Python** (3.6 or later)
- **pip** (Python package installer)

### Setting up the environment
It's recommended to use a virtual environment to avoid conflicts with other Python projects. Here’s how to set up your environment:

1. **Create a virtual environment**:
   - Navigate to your project directory.
   - Run the following command to create a virtual environment:
     ```bash
     python -m venv chatbot_env
     ```
   
2. **Activate the virtual environment**:
   - On Windows:
     ```bash
     chatbot_env\Scripts\activate
     ```
   - On Mac/Linux:
     ```bash
     source chatbot_env/bin/activate
     ```

3. **Install required libraries**:
   After activating the virtual environment, install the required libraries using `pip`:
   ```bash
   pip install -r requirements.txt
   ```
   Alternatively, you can manually install the required packages by running:
   ```bash
   pip install Flask transformers nltk scikit-learn numpy
   ```

4. **Download NLTK data**:
   The application uses the NLTK library for tokenization. Make sure the required NLTK data is downloaded. It will be automatically done during the initial run, but you can manually download the needed data by running:
   ```python
   import nltk
   nltk.download('punkt')
   ```

---

## Running the Application

1. **Ensure that you have the required project files**:
   - `app.py`: Main file containing the Flask app and the chatbot logic.
   - `intents.py`: Contains the intent data (patterns and responses) for the chatbot. 
   - `templates/index.html`: The HTML template for the front-end interface (to interact with the chatbot).
   
   Ensure that these files are in your project directory.

2. **Run the Flask app**:
   In your terminal, run the following command to start the Flask server:
   ```bash
   python app.py
   ```

3. **Access the chatbot**:
   Open a web browser and navigate to `http://127.0.0.1:5000/`. You should see the chatbot's front-end interface, where you can start interacting with the bot.

---

## How the Code Works

### `app.py`:
- **Flask App Setup**: The `Flask` web framework is used to serve the chatbot. The app listens for HTTP requests and sends responses accordingly.
  
- **HuggingFaceChatbot Class**: This class is responsible for initializing the chatbot and handling user interactions.
  - **GPT-2 Model**: The class initializes a GPT-2 model from Hugging Face for generating text. This model is used for text generation (e.g., generating responses based on user input).
  - **NLTK**: The `nltk` library is used for tokenizing the user's input.
  - **Intent Classification**: The bot uses the `TfidfVectorizer` and `MultinomialNB` (Naive Bayes classifier) from `sklearn` to classify user input based on predefined intents (patterns and responses).
  - **Training the Intent Classifier**: The classifier is trained using intent data stored in the `intents.py` file, which contains different queries and corresponding responses.

- **Flask Routes**:
  - `/`: The home route renders the HTML interface (`index.html`).
  - `/get_response`: This route accepts POST requests from the front-end with the user's input, classifies the intent, generates the response, and returns it as a JSON object.

### Flow of Interaction:
1. **User Input**: The user enters a message in the front-end chat interface.
2. **Intent Classification**: The chatbot uses the `TfidfVectorizer` and `MultinomialNB` classifier to determine the intent of the user input.
3. **Response Generation**: Based on the intent, the bot either selects a predefined response or uses the GPT-2 model to generate a response if needed.
4. **Return Response**: The chatbot sends the response back to the front-end, where it is displayed to the user.

---

## Example `intents.py` File Structure:

Here’s a sample structure of how the `intents.py` file may look like. This file defines various patterns (user queries) and their corresponding responses.

```python
intent = {
    "intents": [
        {
            "tag": "greeting",
            "patterns": ["Hi", "Hello", "Good morning", "Hey", "How are you?"],
            "responses": ["Hello! How can we assist you today?", "Hi there! How can I help you?"]
        },
        {
            "tag": "goodbye",
            "patterns": ["Goodbye", "Bye", "See you later"],
            "responses": ["Goodbye! We hope to hear from you soon.", "See you later! Take care."]
        },
        # Add more intents here...
    ]
}
```

---

## Future Enhancements:
- **Improved Intent Recognition**: The chatbot can be extended to recognize more complex intents by training the intent classifier with a larger dataset.
- **Advanced NLP Models**: The GPT-2 model can be fine-tuned for domain-specific queries related to lithium batteries to provide more relevant and accurate responses.
- **Database Integration**: To store conversations, intents, and user interactions for analysis and improvements.

---

## Troubleshooting
- **Module Not Found**: If you encounter issues with missing modules, ensure that all dependencies are installed in the virtual environment using `pip install -r requirements.txt`.
- **Port Already in Use**: If the Flask app fails to start due to the port being used by another application, change the port in the `app.run()` function:
  ```python
  app.run(debug=True, port=5001)
  ```
