let isChatOpen = false;
let currentLanguage = 'en';

function toggleChat() {
    const chatContainer = document.getElementById('chat-container');
    const chatToggle = document.getElementById('chat-toggle');
    isChatOpen = !isChatOpen;
    
    if (isChatOpen) {
        chatContainer.classList.remove('minimized');
        chatToggle.style.display = 'none';
    } else {
        chatContainer.classList.add('minimized');
        chatToggle.style.display = 'block';
    }
}

async function sendMessage() {
    const userInput = document.getElementById("user-input").value.trim();
    if (!userInput) return;

    displayMessage(userInput, "user");
    document.getElementById("user-input").value = "";

    try {
        const requestBody = {
            user_input: userInput,
            target_lang: currentLanguage,
            is_recommendation: isInRecommendationMode,
            recommendation_step: recommendationStep
        };

        const response = await fetch("http://127.0.0.1:5000/get_response", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(requestBody)
        });
        
        const data = await response.json();

        if (isInRecommendationMode) {
            if (data.is_final) {
                displayRecommendation(data.response);
                exitRecommendation();
            } else {
                recommendationStep++;
                if (recommendationStep < recommendationQuestions.length) {
                    displayMessage(recommendationQuestions[recommendationStep], "bot");
                }
            }
        } else {
            const detectedLang = data.detected_language;
            let messageExtra = '';
            if (detectedLang && detectedLang !== currentLanguage) {
                messageExtra = `<div class="detected-language">Original message in ${detectedLang}</div>`;
            }
            displayMessage(data.response + messageExtra, "bot", data.audio_url);
        }
    } catch (error) {
        displayMessage("Sorry, something went wrong.", "bot");
        if (isInRecommendationMode) {
            exitRecommendation();
        }
    }
}

function displayMessage(message, sender, audioUrl = null) {
    const messageDiv = document.createElement("div");
    messageDiv.classList.add("chat-message", sender);

    const messageContainer = document.createElement("div");
    messageContainer.classList.add("message-container");

    if (message.includes('<div class="detected-language">')) {
        const mainMessage = message.split('<div')[0];
        const messageParagraph = document.createElement("p");
        messageParagraph.textContent = mainMessage;
        messageContainer.appendChild(messageParagraph);
        
        const langInfo = document.createElement("div");
        langInfo.innerHTML = '<div' + message.split('<div')[1];
        messageContainer.appendChild(langInfo);
    } else {
        const messageParagraph = document.createElement("p");
        messageParagraph.textContent = message;
        messageContainer.appendChild(messageParagraph);
    }

    if (sender === "bot" && audioUrl) {
        const audioPlayer = document.createElement("audio");
        audioPlayer.classList.add("audio-player");
        audioPlayer.controls = true;
        audioPlayer.src = audioUrl;
        messageContainer.appendChild(audioPlayer);
    }

    messageDiv.appendChild(messageContainer);
    document.getElementById("chat-messages").appendChild(messageDiv);
    document.getElementById("chat-messages").scrollTop = document.getElementById("chat-messages").scrollHeight;
}

function changeLanguage() {
    currentLanguage = document.getElementById('language-select').value;
    if (recognition) {
        recognition.lang = currentLanguage === 'fr' ? 'fr-FR' : 'en-US';
    }
}

// Event Listeners
document.addEventListener('DOMContentLoaded', function() {
    document.getElementById("user-input").addEventListener("keypress", function(event) {
        if (event.key === "Enter") {
            sendMessage();
        }
    });
    
    initSpeechRecognition();
});

document.addEventListener('DOMContentLoaded', function() {
    const chatMessages = document.getElementById('chat-messages');
    const chatInput = document.querySelector('.chat-input-wrapper input');
    const sendButton = document.querySelector('.chat-input-wrapper button');
    const languageSelector = document.querySelector('.language-selector');
    let currentLanguage = 'en';

    function addMessage(text, isUser = false) {
        const messageDiv = document.createElement('div');
        messageDiv.className = isUser ? 'user-message' : 'bot-message';
        messageDiv.textContent = text;
        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    async function sendMessage(message) {
        try {
            const response = await fetch('/get_response', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    user_input: message,
                    target_lang: currentLanguage
                })
            });

            const data = await response.json();
            addMessage(data.response);

            // Play audio if available
            if (data.audio_url) {
                const audio = new Audio(data.audio_url);
                audio.play();
            }
        } catch (error) {
            console.error('Error:', error);
            addMessage('Sorry, an error occurred.');
        }
    }

    // Send message on button click
    sendButton.addEventListener('click', () => {
        const message = chatInput.value.trim();
        if (message) {
            addMessage(message, true);
            sendMessage(message);
            chatInput.value = '';
        }
    });

    // Send message on Enter key
    chatInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            const message = chatInput.value.trim();
            if (message) {
                addMessage(message, true);
                sendMessage(message);
                chatInput.value = '';
            }
        }
    });

    // Language toggle
    languageSelector.addEventListener('click', () => {
        currentLanguage = currentLanguage === 'en' ? 'fr' : 'en';
        languageSelector.textContent = currentLanguage === 'en' ? 'Français' : 'English';
    });
});