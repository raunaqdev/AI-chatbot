let isInRecommendationMode = false;
let recommendationStep = 0;

const recommendationQuestions = [
    "What type of application will you use the battery for? Choose from: portable electronics, vehicles, energy storage, or medical devices",
    "What's your budget constraint? Choose from: low, medium, or high",
    "Do you have size constraints? Answer yes or no",
    "How important is battery lifecycle? Choose from: standard or long"
];

function startBatteryRecommendation() {
    isInRecommendationMode = true;
    recommendationStep = 0;
    
    document.getElementById('rec-mode').style.display = 'flex';
    document.getElementById('rec-button').style.display = 'none';
    
    displayMessage(recommendationQuestions[0], "bot");
}

function exitRecommendation() {
    isInRecommendationMode = false;
    recommendationStep = 0;
    
    document.getElementById('rec-mode').style.display = 'none';
    document.getElementById('rec-button').style.display = 'flex';
    
    displayMessage("Battery recommendation session ended. How else can I help you?", "bot");
}

function displayRecommendation(recommendation) {
    const messageDiv = document.createElement("div");
    messageDiv.classList.add("chat-message", "bot");

    const recommendationCard = document.createElement("div");
    recommendationCard.classList.add("recommendation-card");
    recommendationCard.innerHTML = recommendation.replace(/\n/g, '<br>');

    messageDiv.appendChild(recommendationCard);
    document.getElementById("chat-messages").appendChild(messageDiv);
    document.getElementById("chat-messages").scrollTop = document.getElementById("chat-messages").scrollHeight;
}