from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS
from transformers import AutoModelForCausalLM, AutoTokenizer
from deep_translator import GoogleTranslator
import nltk
import random
import numpy as np
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from gtts import gTTS
import os
import uuid
from datetime import datetime, timedelta
from battery_data import get_battery_recommendation, BATTERY_DATA
from intents import intent

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'
CORS(app)

@app.before_request
def before_request():
    if 'cart' not in session:
        session['cart'] = {}

class BatteryChatbot:
    def __init__(self):
        self.model_name = "gpt2"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        
        nltk.download('punkt', quiet=True)
        self.vectorizer = TfidfVectorizer()
        self.classifier = MultinomialNB()
        
        self.checkout_state = {}
        self.checkout_questions = {
            'name': "Please enter your full name:",
            'email': "Please enter your email address:",
            'phone': "Please enter your phone number:",
            'street': "Please enter your street address:",
            'city': "Please enter your city:",
            'state': "Please enter your state/province:",
            'postal': "Please enter your postal code:",
            'country': "Please enter your country:"
        }
        self.checkout_order = ['name', 'email', 'phone', 'street', 'city', 'state', 'postal', 'country']
        
        self.recommendation_answers = {}
        self.recommendation_state = {
            'active': False,
            'current_step': 0,
            'answers': {}
        }
        
        self.recommendation_questions = [
            "What type of application will you use the battery for? (portable electronics/vehicles/energy storage/medical devices)",
            "What's your budget constraint? (low/medium/high)",
            "Do you have size constraints? (yes/no)",
            "How important is battery lifecycle? (standard/long)"
        ]

        self.patterns, self.responses, self.labels = self._prepare_training_data()
        self._train_intent_classifier()
        self.translator_to_french = None
        self.translator_to_english = None
        self._initialize_translators()
        
    def _initialize_translators(self):
        """Initialize translator instances with retry mechanism"""
        max_retries = 3
        retry_delay = 1  # seconds
        
        for attempt in range(max_retries):
            try:
                self.translator_to_french = GoogleTranslator(source='en', target='fr')
                self.translator_to_english = GoogleTranslator(source='fr', target='en')
                return
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"Failed to initialize translators after {max_retries} attempts: {e}")
                else:
                    time.sleep(retry_delay)
    def _prepare_training_data(self):
        patterns = []
        responses = {}
        labels = []
        
        # Process each intent only once
        seen_tags = set()
        for intent_item in intent["intents"]:
            tag = intent_item["tag"]
            if tag not in seen_tags:
                seen_tags.add(tag)
                patterns.extend([(pattern.lower(), tag) for pattern in intent_item["patterns"]])
                responses[tag] = intent_item["responses"]
                labels.append(tag)
        
        return patterns, responses, labels

    def _classify_intent(self, user_input):
        # Convert input to lowercase for consistent matching
        user_input = user_input.lower().strip()
        
        # First check for exact matches
        for pattern, label in self.patterns:
            if user_input == pattern.lower():
                return label, 1.0
        
        # If no exact match, use TF-IDF and classifier
        X_input = self.vectorizer.transform([user_input])
        intent = self.classifier.predict(X_input)[0]
        confidence = np.max(self.classifier.predict_proba(X_input))
        
        return intent, confidence

    def handle_intent(self, intent, user_input, target_lang='en'):
        """Handle different intents and return appropriate responses"""
        handlers = {
            'greeting': self.handle_greeting,
            'goodbye': lambda u, t: (random.choice(self.responses['goodbye']), self._text_to_speech(random.choice(self.responses['goodbye']), t)),
            'thanks': lambda u, t: (random.choice(self.responses['thanks']), self._text_to_speech(random.choice(self.responses['thanks']), t)),
            'abuse': lambda u, t: (random.choice(self.responses['abuse']), self._text_to_speech(random.choice(self.responses['abuse']), t)),
            'catalog': self.handle_catalog,
            'view_cart': self.handle_view_cart,
            'add_to_cart': self.handle_add_to_cart,
            'remove_from_cart': self.handle_remove_from_cart,
            'clear_cart': self.handle_clear_cart,
            'checkout': self.handle_checkout,
            'help': self.handle_help,
            'product_info': self.handle_product_info,
            'product_inquiry': lambda u, t: (random.choice(self.responses['product_inquiry']), self._text_to_speech(random.choice(self.responses['product_inquiry']), t)),
            'battery_safety': lambda u, t: (random.choice(self.responses['battery_safety']), self._text_to_speech(random.choice(self.responses['battery_safety']), t)),
            'shipping_inquiry': lambda u, t: (random.choice(self.responses['shipping_inquiry']), self._text_to_speech(random.choice(self.responses['shipping_inquiry']), t)),
            'pricing_inquiry': lambda u, t: (random.choice(self.responses['pricing_inquiry']), self._text_to_speech(random.choice(self.responses['pricing_inquiry']), t)),
            'technical_support': lambda u, t: (random.choice(self.responses['technical_support']), self._text_to_speech(random.choice(self.responses['technical_support']), t)),
            'bulk_orders': lambda u, t: (random.choice(self.responses['bulk_orders']), self._text_to_speech(random.choice(self.responses['bulk_orders']), t))
        }
        
        handler = handlers.get(intent)
        if handler:
            try:
                return handler(user_input, target_lang)
            except Exception as e:
                print(f"Handler error for intent {intent}: {str(e)}")
                fallback = "I encountered an error processing your request. Please try again."
                final_response = self._translate_to_french(fallback) if target_lang == 'fr' else fallback
                return final_response, self._text_to_speech(final_response, lang=target_lang)
        
        # Default response for unknown intents
        response = random.choice(self.responses.get(intent, ["I didn't understand that. Could you please rephrase?"]))
        final_response = self._translate_to_french(response) if target_lang == 'fr' else response
        return final_response, self._text_to_speech(final_response, lang=target_lang)
    def _prepare_training_data(self):
        """Prepare training data from intents"""
        patterns = []
        responses = {}
        labels = []

        for intent_item in intent["intents"]:
            tag = intent_item["tag"]
            patterns.extend([(pattern, tag) for pattern in intent_item["patterns"]])
            responses[tag] = intent_item["responses"]
            labels.append(tag)

        return patterns, responses, labels

    def _translate_to_french(self, text):
        """Translate text to French with improved error handling"""
        if not text:
            return text
            
        try:
            # Reinitialize translator if needed
            if not self.translator_to_french:
                self._initialize_translators()
                
            # Split long text into chunks if needed (Google Translator has character limits)
            max_chunk_size = 4000
            if len(text) > max_chunk_size:
                chunks = [text[i:i + max_chunk_size] for i in range(0, len(text), max_chunk_size)]
                translated_chunks = []
                for chunk in chunks:
                    translated_chunk = self.translator_to_french.translate(chunk)
                    if translated_chunk:
                        translated_chunks.append(translated_chunk)
                    time.sleep(0.5)  # Add delay between chunks to avoid rate limiting
                return ' '.join(translated_chunks)
            else:
                return self.translator_to_french.translate(text)
        except Exception as e:
            print(f"Translation to French failed: {e}")
            return text  # Return original text if translation fails
    
    def _translate_to_english(self, text):
        """Translate text to English with improved error handling"""
        if not text:
            return text
            
        try:
            # Reinitialize translator if needed
            if not self.translator_to_english:
                self._initialize_translators()
                
            # Split long text into chunks if needed
            max_chunk_size = 4000
            if len(text) > max_chunk_size:
                chunks = [text[i:i + max_chunk_size] for i in range(0, len(text), max_chunk_size)]
                translated_chunks = []
                for chunk in chunks:
                    translated_chunk = self.translator_to_english.translate(chunk)
                    if translated_chunk:
                        translated_chunks.append(translated_chunk)
                    time.sleep(0.5)  # Add delay between chunks
                return ' '.join(translated_chunks)
            else:
                return self.translator_to_english.translate(text)
        except Exception as e:
            print(f"Translation to English failed: {e}")
            return text  # Return original text if translation fails

    def _train_intent_classifier(self):
        """Train the intent classifier with the prepared data"""
        X, y = zip(*self.patterns)
        X_tfidf = self.vectorizer.fit_transform(X)
        self.classifier.fit(X_tfidf, y)
        
        # Print some debugging info
        print(f"Trained on {len(X)} patterns across {len(set(y))} intents")

    def _classify_intent(self, user_input):
        """Classify the intent of user input"""
        user_input = user_input.lower().strip()
        
        # Check for exact matches first
        for pattern, label in self.patterns:
            if user_input == pattern.lower().strip():
                return label, 1.0
        
        # If no exact match, use the classifier
        try:
            X_input = self.vectorizer.transform([user_input])
            intent = self.classifier.predict(X_input)[0]
            confidence = np.max(self.classifier.predict_proba(X_input))
            
            # Print debugging info
            print(f"Classified '{user_input}' as '{intent}' with confidence {confidence}")
            
            return intent, confidence
        except Exception as e:
            print(f"Classification error: {str(e)}")
            return 'unknown', 0.0
    def _text_to_speech(self, text, lang='en'):
        try:
            filename = f"response_{str(uuid.uuid4())[:8]}.mp3"
            filepath = os.path.join('static/audio', filename)
            tts = gTTS(text=text, lang=lang)
            tts.save(filepath)
            return filename
        except Exception as e:
            print(f"Text-to-speech error: {e}")
            return None


    def handle_greeting(self, user_input, target_lang):
        response = random.choice(self.responses['greeting'])
        final_response = self._translate_to_french(response) if target_lang == 'fr' else response
        return final_response, self._text_to_speech(final_response, lang=target_lang)

    def handle_catalog(self, user_input, target_lang):
        catalog_response = self.format_product_catalog()
        final_response = self._translate_to_french(catalog_response) if target_lang == 'fr' else catalog_response
        return final_response, None
    def _format_recommendations(self, recommendations):
        """Format battery recommendations for display"""
        if not recommendations:
            return "No recommendations found."
            
        response = "🔋 Based on your requirements, here are my top recommendations:\n\n"
        
        for i, rec in enumerate(recommendations, 1):
            response += f"""
    Recommendation #{i}: {rec['name']} ({rec['model']})
    ━━━━━━━━━━━━━━━━━━━━
    • Manufacturer: {rec['manufacturer']}
    • Price: ${rec['price']}
    • Capacity: {rec['capacity']}
    • Stock: {rec['stock']} units

    Key Features:
    • Energy Density: {rec['features']['energy_density']}
    • Cycle Life: {rec['features']['cycle_life']}
    • Maintenance: {rec['features']['maintenance']}

    Advantages:
    • {', '.join(rec['advantages'][:2])}

    Type 'add {rec['model']} to cart' to purchase
    """
            
            response += "\nWould you like more details about any of these batteries?"
            return response
    def handle_recommendation(self, user_input, step):
        """
        Handles the battery recommendation process, collecting user input
        and providing a final recommendation.
        """
        if step < len(self.recommendation_questions):
            # Save the user's input for the current step
            if step == 0:
                self.recommendation_answers['application'] = user_input.lower()
            elif step == 1:
                self.recommendation_answers['budget'] = user_input.lower()
            elif step == 2:
                self.recommendation_answers['size_constraint'] = 'small' if user_input.lower() == 'yes' else 'standard'
            elif step == 3:
                self.recommendation_answers['lifecycle'] = user_input.lower()

            # Provide the next question
            if step + 1 < len(self.recommendation_questions):
                next_question = self.recommendation_questions[step + 1]
                return next_question, None

            # All inputs collected, process the recommendation
            recommendations = get_battery_recommendation(self.recommendation_answers)
            if not recommendations:
                response = "I couldn't find any batteries matching your requirements."
            else:
                response = "Here are the recommended batteries based on your inputs:\n"
                for i, rec in enumerate(recommendations, 1):
                    response += f"\nOption {i}: {rec['name']} ({rec['model']})\n"
                    response += f"  - Manufacturer: {rec['manufacturer']}\n"
                    response += f"  - Price: ${rec['price']}\n"
                    response += f"  - Capacity: {rec['capacity']}\n"
                    response += f"  - Stock: {rec['stock']} units\n"
                    response += f"  - Features: {', '.join(f'{k}: {v}' for k, v in rec['features'].items())}\n"
                    response += f"  - Advantages: {', '.join(rec['advantages'])}\n"
                    response += f"  - Limitations: {', '.join(rec['limitations'])}\n"
                    response += f"  - Bulk Discounts: {rec['bulk_discounts']}\n"

            # Reset recommendation answers for next use
            self.recommendation_answers = {}
            return response, None
        else:
            return "Invalid step in recommendation process.", None

    
    
    def handle_view_cart(self, user_input, target_lang):
        cart_contents = self.format_cart_contents()
        final_response = self._translate_to_french(cart_contents) if target_lang == 'fr' else cart_contents
        return final_response, self._text_to_speech(final_response, lang=target_lang)

    def handle_add_to_cart(self, user_input, target_lang):
        model = self.extract_model_from_input(user_input)
        if model:
            if model in session['cart']:
                session['cart'][model] += 1
            else:
                session['cart'][model] = 1
            session.modified = True
            response = f"✅ Added {model} to your cart."
        else:
            response = "❌ I couldn't find that product. Please specify a valid model number."
        
        final_response = self._translate_to_french(response) if target_lang == 'fr' else response
        return final_response, self._text_to_speech(final_response, lang=target_lang)

    def handle_remove_from_cart(self, user_input, target_lang):
        model = self.extract_model_from_input(user_input)
        if model and model in session['cart']:
            del session['cart'][model]
            session.modified = True
            response = f"Removed {model} from your cart."
        else:
            response = "I couldn't find that item in your cart."
        
        final_response = self._translate_to_french(response) if target_lang == 'fr' else response
        return final_response, self._text_to_speech(final_response, lang=target_lang)

    def handle_clear_cart(self, user_input, target_lang):
        session['cart'] = {}
        session.modified = True
        response = "Your cart has been cleared."
        final_response = self._translate_to_french(response) if target_lang == 'fr' else response
        return final_response, self._text_to_speech(final_response, lang=target_lang)

    def handle_checkout(self, user_input, target_lang):
        if not session.get('cart'):
            response = "Your cart is empty. Please add items before checking out."
            final_response = self._translate_to_french(response) if target_lang == 'fr' else response
            return final_response, self._text_to_speech(final_response, lang=target_lang)
        
        self.checkout_state = {
            'active': True,
            'current_step': 0,
            'data': {}
        }
        
        response = f"{self.format_cart_contents()}\n\nPlease enter your full name:"
        final_response = self._translate_to_french(response) if target_lang == 'fr' else response
        return final_response, self._text_to_speech(final_response, lang=target_lang)

    def handle_help(self, user_input, target_lang):
            response = """
        Here's how I can help you:
        • View products: 'show catalog'
        • Get recommendations: 'recommend a battery'
        • Cart commands: 'show cart', 'add to cart', 'remove from cart'
        • Checkout: 'checkout'
        • Product info: 'tell me about [product model]'
        """
            final_response = self._translate_to_french(response) if target_lang == 'fr' else response
            return final_response, self._text_to_speech(final_response, lang=target_lang)
            # Add these methods to your BatteryChatbot class

    def handle_product_info(self, user_input, target_lang):
        """Handle product information requests"""
        model = self.extract_model_from_input(user_input)
        if model:
            product_info = self.get_product_details(model)
            if product_info:
                final_response = self._translate_to_french(product_info) if target_lang == 'fr' else product_info
                return final_response, self._text_to_speech(final_response, lang=target_lang)
        
        response = "I couldn't find information about that product. Please specify a valid model number."
        final_response = self._translate_to_french(response) if target_lang == 'fr' else response
        return final_response, self._text_to_speech(final_response, lang=target_lang)

    def get_product_details(self, model):
        """Get detailed product information"""
        for battery_type, battery_data in BATTERY_DATA.items():
            for product in battery_data['products']:
                if product['model'] == model:
                    return f"""
    🔋 Product Details for {battery_data['name']} ({product['model']})
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    Manufacturer: {product['manufacturer']}
    Price: ${product['price']}
    Stock: {product['stock']} units

    Technical Specifications:
    • Capacity: {product['capacity']}
    • Voltage: {product['voltage']}
    • Weight: {product['weight']}
    • Dimensions: {product['dimensions']}

    Features:
    • Energy Density: {battery_data['features']['energy_density']}
    • Cycle Life: {battery_data['features']['cycle_life']}
    • Maintenance: {battery_data['features']['maintenance']}

    Applications: {', '.join(battery_data['applications'][:3])}

    Certifications: {', '.join(product['certifications'])}
    Warranty: {product['warranty']}

    Type 'add {product['model']} to cart' to purchase
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    """
        return None

    def format_product_catalog(self):
        """Format the full product catalog"""
        catalog = "🏪 Our Battery Catalog:\n\n"
        for battery_type, battery_data in BATTERY_DATA.items():
            for product in battery_data['products']:
                catalog += f"""
    🔋 {battery_data['name']} ({product['model']})
    Price: ${product['price']}
    Capacity: {product['capacity']}
    Stock: {product['stock']} units
    Type 'details {product['model']}' for more information
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    """
        return catalog

    def handle_checkout_response(self, user_input):
        """Handle responses during checkout process"""
        current_field = self.checkout_order[self.checkout_state['current_step']]
        self.checkout_state['data'][current_field] = user_input
        
        # Move to next question
        self.checkout_state['current_step'] += 1
        
        # Check if checkout is complete
        if self.checkout_state['current_step'] >= len(self.checkout_order):
            # Process the completed checkout
            summary = self.complete_checkout()
            return summary, None
        
        # Get next question
        next_field = self.checkout_order[self.checkout_state['current_step']]
        return self.checkout_questions[next_field], None

    def complete_checkout(self):
        """Complete the checkout process"""
        transaction_id = str(uuid.uuid4())
        order_data = self.checkout_state['data']
        cart_contents = self.format_cart_contents()
        
        summary = f"""
    🎉 Order Successfully Placed!

    Order ID: {transaction_id}

    {cart_contents}

    📦 Shipping to:
    {order_data['name']}
    {order_data['street']}
    {order_data['city']}, {order_data['state']} {order_data['postal']}
    {order_data['country']}

    📱 Contact:
    Email: {order_data['email']}
    Phone: {order_data['phone']}

    Thank you for your purchase! You will receive a confirmation email shortly.
    """
        
        # Clear checkout state and cart
        self.checkout_state = {}
        session['cart'] = {}
        session.modified = True
        
        return summary

    # Update the handle_intent method to match the available handlers

    def format_cart_contents(self):
        if not session['cart']:
            return "🛒 Your cart is empty."
        
        cart_display = "🛒 Your Shopping Cart:\n"
        total = 0
        
        for model, quantity in session['cart'].items():
            for battery_type, battery_data in BATTERY_DATA.items():
                for product in battery_data['products']:
                    if product['model'] == model:
                        subtotal = quantity * product['price']
                        total += subtotal
                        cart_display += f"\n{battery_data['name']} ({model})"
                        cart_display += f"\n• Quantity: {quantity}"
                        cart_display += f"\n• Price: ${product['price']} each"
                        cart_display += f"\n• Subtotal: ${subtotal}\n"
        
        cart_display += f"\n💰 Total: ${total}"
        return cart_display

    def extract_model_from_input(self, user_input):
        words = user_input.lower().split()
        for word in words:
            for battery_type, battery_data in BATTERY_DATA.items():
                for product in battery_data['products']:
                    if product['model'].lower() == word:
                        return product['model']
        return None

    def _format_bullet_points(self, items):
        """Helper method to format bullet points"""
        return '\n'.join(f"• {item}" for item in items[:3])  # Show top 3 advantages
    def respond(self, user_input, target_lang='en'):
        try:
            # If in recommendation mode, process the current step
            if self.recommendation_answers.get('active'):
                step = self.recommendation_answers.get('current_step', 0)
                response, _ = self.handle_recommendation(user_input, step)
                if step < len(self.recommendation_questions) - 1:
                    self.recommendation_answers['current_step'] += 1
                else:
                    self.recommendation_answers['active'] = False
                return response, None

            # Normal chatbot response handling
            processed_input = self._translate_to_english(user_input) if target_lang == 'fr' else user_input
            intent, confidence = self._classify_intent(processed_input)

            if intent == 'recommendation':
                # Start recommendation flow
                self.recommendation_answers = {'active': True, 'current_step': 0}
                return self.recommendation_questions[0], None

            # Handle other intents
            return self.handle_intent(intent, processed_input, target_lang)

        except Exception as e:
            print(f"Error in respond method: {e}")
            fallback = "Sorry, an error occurred. Please try again."
            final_response = self._translate_to_french(fallback) if target_lang == 'fr' else fallback
            return final_response, None

chatbot = BatteryChatbot()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_response', methods=['POST'])
def get_response():
    data = request.get_json()
    user_input = data.get('user_input')
    target_lang = data.get('target_lang', 'en')
    
    try:
        response_text, audio_filename = chatbot.respond(user_input, target_lang)
        return jsonify({
            'response': response_text,
            'audio_url': f'/static/audio/{audio_filename}' if audio_filename else None
        })
    except Exception as e:
        return jsonify({
            'response': "Sorry, an error occurred.",
            'error': str(e)
        }), 500

@app.route('/clear_cart', methods=['POST'])
def clear_cart():
    try:
        session['cart'] = {}
        return jsonify({'message': 'Cart cleared successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def not_found_error(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == "__main__":
    app.run(debug=True)