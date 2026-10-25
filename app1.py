# app.py
from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import os
import json
import re
from datetime import datetime

app = Flask(__name__, static_url_path='/static')

# Load data and model (in a real application, you would have actual models)
# For this example, we'll use simple dummy data and logic
CROP_DATA = {
    "rice": {
        "avg_yield": "3.5 tonnes/hectare",
        "price_range": "₹1800-2200/quintal",
        "growing_season": "Kharif (June-November)",
        "water_req": "High"
    },
    "wheat": {
        "avg_yield": "3.2 tonnes/hectare",
        "price_range": "₹1900-2400/quintal",
        "growing_season": "Rabi (November-April)",
        "water_req": "Medium"
    },
    "maize": {
        "avg_yield": "2.8 tonnes/hectare", 
        "price_range": "₹1650-2100/quintal",
        "growing_season": "Both Kharif and Rabi",
        "water_req": "Medium"
    },
    "cotton": {
        "avg_yield": "450-500 kg/hectare",
        "price_range": "₹5500-6500/quintal",
        "growing_season": "Kharif (May-November)",
        "water_req": "Medium"
    },
    "sugarcane": {
        "avg_yield": "70-80 tonnes/hectare",
        "price_range": "₹275-325/quintal",
        "growing_season": "Year-round (plant crop: 12-18 months)",
        "water_req": "High"
    }
}

# Load agricultural institutes data (dummy data)
AG_INSTITUTES = {
    "Maharashtra": [
        {"name": "Krishi Vigyan Kendra, Pune", "phone": "020-25654239", "address": "KVK Pune, Maharashtra"},
        {"name": "Maharashtra Agricultural University", "phone": "022-28748976", "address": "MAU Campus, Mumbai"}
    ],
    "Punjab": [
        {"name": "Punjab Agricultural University", "phone": "0161-2401960", "address": "PAU Campus, Ludhiana"},
        {"name": "KVK Amritsar", "phone": "0183-2258872", "address": "KVK Campus, Amritsar"}
    ],
    "Uttar Pradesh": [
        {"name": "Chandra Shekhar Azad University of Agriculture", "phone": "0512-2534156", "address": "CSAUA Campus, Kanpur"},
        {"name": "KVK Lucknow", "phone": "0522-2668700", "address": "KVK Campus, Lucknow"}
    ],
    "Karnataka": [
        {"name": "University of Agricultural Sciences", "phone": "080-23330153", "address": "UAS Campus, Bangalore"},
        {"name": "KVK Mysore", "phone": "0821-2362146", "address": "KVK Campus, Mysore"}
    ],
    "default": [
        {"name": "Indian Council of Agricultural Research", "phone": "011-25841490", "address": "ICAR Headquarters, New Delhi"},
        {"name": "National Agricultural Helpline", "phone": "1800-180-1551", "address": "Available across India"}
    ]
}

# Common agricultural problems and solutions database
COMMON_ISSUES = {
    "pest": {
        "response": "Based on your description, it sounds like a pest infestation. Here are some organic and chemical control measures you can try:",
        "solutions": [
            "Neem oil spray (Mix 5ml neem oil with 1L water and spray weekly)",
            "Introduce beneficial insects like ladybugs or lacewings",
            "For severe infestations, consult with local agricultural extension about appropriate pesticides"
        ]
    },
    "disease": {
        "response": "This appears to be a crop disease. Here are some management strategies:",
        "solutions": [
            "Remove and destroy infected plant parts",
            "Improve air circulation between plants",
            "Apply copper-based fungicide for fungal diseases",
            "Rotate crops in the next season to break disease cycles"
        ]
    },
    "water": {
        "response": "You seem to be experiencing irrigation or water-related issues. Consider these approaches:",
        "solutions": [
            "Implement drip irrigation for water conservation",
            "Add organic matter to soil to improve water retention",
            "Create proper drainage channels if waterlogging is the issue",
            "Consider rainwater harvesting techniques for your farm"
        ]
    },
    "soil": {
        "response": "Your description suggests soil fertility or quality issues. Try these solutions:",
        "solutions": [
            "Get your soil tested to understand exact deficiencies",
            "Apply appropriate organic amendments (compost, vermicompost)",
            "Consider green manuring with legumes to improve nitrogen content",
            "Follow balanced fertilization based on soil test results"
        ]
    },
    "yield": {
        "response": "To improve crop yield, consider these strategies:",
        "solutions": [
            "Use quality seeds from reliable sources",
            "Follow recommended spacing and planting techniques",
            "Implement timely irrigation and fertilizer application",
            "Adopt integrated pest management practices"
        ]
    }
}

# Intent recognition patterns
INTENT_PATTERNS = {
    'greeting': r'\b(hi|hello|greetings|namaste|hey)\b',
    'yield_prediction': r'\b(yield|production|harvest|output)\b.*\b(prediction|forecast|estimate)\b',
    'price_prediction': r'\b(price|rate|cost|value)\b.*\b(prediction|forecast|estimate)\b',
    'crop_info': r'\b(information|details|data|facts)\b.*\b(crop|plant|cultivation)\b',
    'pest_disease': r'\b(pest|disease|infection|bug|insect)\b',
    'water_irrigation': r'\b(water|irrigation|moisture|rain|drought)\b',
    'soil_issue': r'\b(soil|nutrient|fertility|ph)\b',
    'weather': r'\b(weather|climate|temperature|forecast|rain)\b',
    'market': r'\b(market|sell|buying|trading|mandi)\b',
    'general_help': r'\b(help|assist|support|guidance)\b',
    'location': r'\b(connect|institute|expert|specialist|advisor)\b'
}

# Chat history storage
class ChatHistory:
    def __init__(self):
        self.conversations = {}
    
    def add_message(self, session_id, role, message):
        if session_id not in self.conversations:
            self.conversations[session_id] = []
        
        self.conversations[session_id].append({
            "role": role,
            "message": message,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
    
    def get_conversation(self, session_id):
        return self.conversations.get(session_id, [])
    
    def clear_conversation(self, session_id):
        if session_id in self.conversations:
            del self.conversations[session_id]

chat_history = ChatHistory()

# Functions for predictions and recommendations
def predict_crop_yield(crop, area, soil_type, rainfall):
    """Dummy function to predict crop yield"""
    if crop.lower() in CROP_DATA:
        base_yield = CROP_DATA[crop.lower()]["avg_yield"]
        return f"Based on your inputs, the estimated yield for {crop} would be approximately {base_yield}."
    else:
        return f"Sorry, I don't have yield data for {crop}. Please try another crop or contact your local agricultural institute for more information."

def predict_crop_price(crop, season):
    """Dummy function to predict crop price"""
    if crop.lower() in CROP_DATA:
        price_range = CROP_DATA[crop.lower()]["price_range"]
        return f"For the {season} season, the expected price range for {crop} is approximately {price_range}."
    else:
        return f"Sorry, I don't have price data for {crop}. Please try another crop or check with the local market committee."

def get_crop_recommendation(soil_type, rainfall, season):
    """Dummy function for crop recommendations"""
    if rainfall > 1000:  # high rainfall
        if season.lower() == "kharif":
            return "Based on your soil type and high rainfall, rice would be a good option for the Kharif season."
        else:
            return "For the Rabi season with your conditions, wheat or mustard could be suitable options."
    else:  # moderate to low rainfall
        if season.lower() == "kharif":
            return "With moderate rainfall, consider maize, cotton, or pulses for the Kharif season."
        else:
            return "For Rabi season, barley, gram, or safflower might be suitable for your conditions."

def detect_intent(message):
    """Detect the primary intent from the user message"""
    for intent, pattern in INTENT_PATTERNS.items():
        if re.search(pattern, message, re.IGNORECASE):
            return intent
    return "unknown"

def extract_location(message):
    """Extract location information from message - simplified version"""
    # In a real application, use NER (Named Entity Recognition) for better extraction
    states = ["maharashtra", "punjab", "uttar pradesh", "karnataka", 
              "tamil nadu", "kerala", "gujarat", "rajasthan", "bihar",
              "west bengal", "odisha", "telangana", "andhra pradesh"]
    
    message_lower = message.lower()
    for state in states:
        if state in message_lower:
            return state
    
    # Look for location phrases
    location_patterns = [
        r'from\s+(\w+\s+\w+|\w+)',
        r'in\s+(\w+\s+\w+|\w+)',
        r'at\s+(\w+\s+\w+|\w+)'
    ]
    
    for pattern in location_patterns:
        match = re.search(pattern, message_lower)
        if match:
            location = match.group(1)
            if any(state in location for state in states):
                return location
    
    return None

def identify_issue_type(message):
    """Identify the type of agricultural issue from message"""
    issue_keywords = {
        "pest": ["pest", "insect", "bug", "caterpillar", "aphid", "worm", "beetle"],
        "disease": ["disease", "rust", "blight", "rot", "mildew", "wilt", "spot", "yellowing"],
        "water": ["water", "irrigation", "moisture", "drought", "flood", "dry", "rain"],
        "soil": ["soil", "nutrient", "fertility", "ph", "deficiency", "acidic", "alkaline"],
        "yield": ["yield", "production", "growth", "small", "not growing", "stunted"]
    }
    
    message_lower = message.lower()
    
    for issue_type, keywords in issue_keywords.items():
        if any(keyword in message_lower for keyword in keywords):
            return issue_type
    
    return None

def generate_response(message, session_id):
    """Generate appropriate response based on user message"""
    # Detect intent
    intent = detect_intent(message)
    
    # Extract location from message if present
    location = extract_location(message)
    
    # Record this message
    chat_history.add_message(session_id, "user", message)
    
    # Check if we need location but don't have it
    location_needed = False
    response = ""
    
    # Generate response based on intent
    if intent == "greeting":
        response = ("Hello! I'm your agricultural assistant. I can help with crop yield predictions, "
                   "price forecasts, pest management, and connect you with agricultural institutes. "
                   "How can I assist you today?")
    
    elif intent == "yield_prediction":
        if "rice" in message.lower():
            response = predict_crop_yield("rice", "1 hectare", "loamy", 1200)
        elif "wheat" in message.lower():
            response = predict_crop_yield("wheat", "1 hectare", "clay", 800)
        elif "maize" in message.lower():
            response = predict_crop_yield("maize", "1 hectare", "sandy loam", 900)
        else:
            response = ("I can provide yield predictions for rice, wheat, maize, cotton and sugarcane. "
                       "Please specify which crop you're interested in and, if possible, your land area.")
    
    elif intent == "price_prediction":
        if "rice" in message.lower():
            response = predict_crop_price("rice", "current")
        elif "wheat" in message.lower():
            response = predict_crop_price("wheat", "current")
        elif "maize" in message.lower():
            response = predict_crop_price("maize", "current")
        elif "cotton" in message.lower():
            response = predict_crop_price("cotton", "current")
        elif "sugarcane" in message.lower():
            response = predict_crop_price("sugarcane", "current")
        else:
            response = ("I can provide price predictions for rice, wheat, maize, cotton and sugarcane. "
                       "Please specify which crop you're interested in.")
    
    elif intent == "crop_info":
        for crop in CROP_DATA:
            if crop in message.lower():
                data = CROP_DATA[crop]
                response = (f"Information about {crop}:\n"
                           f"- Average yield: {data['avg_yield']}\n"
                           f"- Price range: {data['price_range']}\n"
                           f"- Growing season: {data['growing_season']}\n"
                           f"- Water requirement: {data['water_req']}")
                break
        
        if not response:
            response = ("I can provide information about rice, wheat, maize, cotton, and sugarcane. "
                       "Which crop would you like to learn more about?")
    
    elif intent in ["pest_disease", "water_irrigation", "soil_issue"]:
        # Identify specific issue
        issue_type = identify_issue_type(message)
        
        if issue_type and issue_type in COMMON_ISSUES:
            issue_data = COMMON_ISSUES[issue_type]
            response = issue_data["response"] + "\n"
            response += "\n".join([f"- {sol}" for sol in issue_data["solutions"]])
            
            # Add a note about connecting to experts for serious issues
            response += ("\n\nIf this issue persists, I can connect you with agricultural experts. "
                        "Just let me know your location (state/district).")
        else:
            response = ("I understand you're facing a farming issue. To provide specific advice, "
                       "could you please describe the symptoms or problems in more detail?")
    
    elif intent == "weather":
        response = ("For accurate weather forecasts and agricultural advisories based on weather, "
                   "I recommend checking the official Indian Meteorological Department website or app. "
                   "Alternatively, I can connect you with local agricultural experts who provide weather-based advisories. "
                   "Would you like me to connect you? If yes, please share your location.")
        location_needed = True
    
    elif intent == "market":
        response = ("For current market rates and trends, you can check the official e-NAM (National Agriculture Market) portal. "
                   "I can also connect you with local market committee or agricultural marketing board for specific information. "
                   "Would you like me to connect you? If yes, please share your location.")
        location_needed = True
    
    elif intent == "location" or intent == "general_help":
        location_needed = True
        if location:
            # Normalize state name for dictionary lookup
            state = location.title()
            if state.lower() == "up":
                state = "Uttar Pradesh"
            
            # Get institutes for the state or use default
            institutes = AG_INSTITUTES.get(state, AG_INSTITUTES["default"])
            
            response = f"Here are agricultural institutes in or serving {state} that can help you:\n"
            for institute in institutes:
                response += f"\n- {institute['name']}\n  Phone: {institute['phone']}\n  Address: {institute['address']}"
                
            response += ("\n\nYou can contact them for personalized guidance. Would you like information "
                        "about any specific agricultural issue in the meantime?")
        else:
            response = ("I'd be happy to connect you with agricultural experts or institutes. "
                       "Could you please share your state or district so I can find the nearest resource?")
    
    else:
        response = ("I'm not sure I understand your query. I can help with crop yield predictions, "
                   "price forecasts, pest management advice, weather implications, or connect you with "
                   "agricultural institutes. How can I assist you today?")
    
    # If we need location but don't have it, append a request
    if location_needed and not location and "location" not in response.lower():
        response += "\n\nTo provide more specific assistance, could you please share your location (state/district)?"
    
    # Record the response
    chat_history.add_message(session_id, "bot", response)
    
    return response

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message', '')
    session_id = request.json.get('session_id', 'default')
    
    response = generate_response(user_message, session_id)
    
    return jsonify({
        'reply': response
    })

@app.route('/history', methods=['GET'])
def get_history():
    session_id = request.args.get('session_id', 'default')
    history = chat_history.get_conversation(session_id)
    return jsonify(history)

@app.route('/clear', methods=['POST'])
def clear_chat():
    session_id = request.json.get('session_id', 'default')
    chat_history.clear_conversation(session_id)
    return jsonify({'status': 'success'})

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    if not os.path.exists('templates'):
        os.makedirs('templates')
    
    # Create template file
    with open('templates/index.html', 'w') as f:
        f.write('''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Crop Advisor Assistant</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f5f5f5;
            color: #333;
        }
        .container {
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }
        .header {
            background-color: #4CAF50;
            color: white;
            padding: 20px;
            text-align: center;
            border-radius: 8px 8px 0 0;
        }
        .chat-container {
            height: 500px;
            background-color: white;
            overflow-y: auto;
            padding: 20px;
            border-left: 1px solid #ddd;
            border-right: 1px solid #ddd;
        }
        .input-container {
            display: flex;
            padding: 15px;
            background-color: #f8f8f8;
            border: 1px solid #ddd;
            border-radius: 0 0 8px 8px;
        }
        input {
            flex: 1;
            padding: 12px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 16px;
        }
        button {
            background-color: #4CAF50;
            color: white;
            border: none;
            padding: 12px 20px;
            margin-left: 10px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 16px;
        }
        button:hover {
            background-color: #45a049;
        }
        .user-message {
            background-color: #e6f7ff;
            padding: 10px 15px;
            border-radius: 18px 18px 0 18px;
            margin: 10px 0;
            max-width: 70%;
            align-self: flex-end;
            margin-left: auto;
            word-wrap: break-word;
        }
        .bot-message {
            background-color: #f1f1f1;
            padding: 10px 15px;
            border-radius: 18px 18px 18px 0;
            margin: 10px 0;
            max-width: 70%;
            word-wrap: break-word;
            white-space: pre-line;
        }
        .message-container {
            display: flex;
            flex-direction: column;
        }
        .suggestion-chips {
            display: flex;
            flex-wrap: wrap;
            margin: 10px 0;
        }
        .suggestion-chip {
            background-color: #e0e0e0;
            padding: 8px 15px;
            border-radius: 16px;
            margin: 5px;
            font-size: 14px;
            cursor: pointer;
            transition: background-color 0.3s;
        }
        .suggestion-chip:hover {
            background-color: #d0d0d0;
        }
        .clear-button {
            background-color: #f44336;
            margin-top: 10px;
        }
        .clear-button:hover {
            background-color: #e53935;
        }
        .typing {
            font-style: italic;
            color: #888;
            margin: 10px 0;
        }
        @media (max-width: 600px) {
            .container {
                padding: 10px;
            }
            .user-message, .bot-message {
                max-width: 85%;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Crop Advisor Assistant</h1>
            <p>Your AI companion for agricultural queries</p>
        </div>
        <div class="chat-container" id="chatContainer"></div>
        <div class="suggestion-chips" id="suggestionChips">
            <div class="suggestion-chip" onclick="sendSuggestion('Hello, how can you help me?')">How can you help?</div>
            <div class="suggestion-chip" onclick="sendSuggestion('What will be the rice yield this season?')">Rice yield prediction</div>
            <div class="suggestion-chip" onclick="sendSuggestion('What is the current price of wheat?')">Wheat prices</div>
            <div class="suggestion-chip" onclick="sendSuggestion('My crop has yellow spots on leaves')">Pest/disease help</div>
            <div class="suggestion-chip" onclick="sendSuggestion('Connect me to agricultural experts')">Connect to experts</div>
        </div>
        <div class="input-container">
            <input type="text" id="userInput" placeholder="Type your agricultural query here..." onkeypress="handleKeyPress(event)">
            <button onclick="sendMessage()">Send</button>
        </div>
        <button class="clear-button" onclick="clearChat()">Clear Chat</button>
    </div>

    <script>
        // Generate a random session ID for this chat
        const sessionId = Math.random().toString(36).substring(2, 15);
        let isTyping = false;

        window.onload = function() {
            // Add initial bot message
            addBotMessage("Hello! I'm your agricultural assistant. I can help with crop yield predictions, price forecasts, pest management, and connect you with agricultural institutes. How can I assist you today?");
        }

        function handleKeyPress(event) {
            if (event.key === 'Enter') {
                sendMessage();
            }
        }

        function sendSuggestion(text) {
            document.getElementById('userInput').value = text;
            sendMessage();
        }

        function sendMessage() {
            const userInput = document.getElementById('userInput');
            const message = userInput.value.trim();
            
            if (message === '') return;
            
            // Add user message to chat
            addUserMessage(message);
            userInput.value = '';
            
            // Show typing indicator
            showTypingIndicator();
            
            // Send message to server
            fetch('/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    message: message,
                    session_id: sessionId
                }),
            })
            .then(response => response.json())
            .then(data => {
                // Remove typing indicator
                hideTypingIndicator();
                
                // Add bot response to chat
                addBotMessage(data.reply);
            })
            .catch(error => {
                hideTypingIndicator();
                addBotMessage("Sorry, I'm having trouble connecting. Please try again later.");
                console.error('Error:', error);
            });
        }

        function showTypingIndicator() {
            if (!isTyping) {
                isTyping = true;
                const chatContainer = document.getElementById('chatContainer');
                const typingDiv = document.createElement('div');
                typingDiv.id = 'typingIndicator';
                typingDiv.className = 'typing';
                typingDiv.textContent = 'Crop Assistant is typing...';
                chatContainer.appendChild(typingDiv);
                chatContainer.scrollTop = chatContainer.scrollHeight;
            }
        }

        function hideTypingIndicator() {
            isTyping = false;
            const typingIndicator = document.getElementById('typingIndicator');
            if (typingIndicator) {
                typingIndicator.remove();
            }
        }

        function addUserMessage(message) {
            const chatContainer = document.getElementById('chatContainer');
            const messageElement = document.createElement('div');
            messageElement.className = 'user-message';
            messageElement.textContent = message;
            chatContainer.appendChild(messageElement);
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }

        function addBotMessage(message) {
            const chatContainer = document.getElementById('chatContainer');
            const messageElement = document.createElement('div');
            messageElement.className = 'bot-message';
            messageElement.textContent = message;
            chatContainer.appendChild(messageElement);
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }

        function clearChat() {
            const chatContainer = document.getElementById('chatContainer');
            chatContainer.innerHTML = '';
            
            // Send clear request to server
            fetch('/clear', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    session_id: sessionId
                }),
            })
            .then(response => response.json())
            .then(data => {
                addBotMessage("Hello! I'm your agricultural assistant. I can help with crop yield predictions, price forecasts, pest management, and connect you with agricultural institutes. How can I assist you today?");
            })
            .catch(error => {
                console.error('Error:', error);
            });
        }
    </script>
</body>
</html>
        ''')
    
    app.run(debug=True)