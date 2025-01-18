let recognition = null;
let isRecording = false;

function initSpeechRecognition() {
    if ('webkitSpeechRecognition' in window) {
        recognition = new webkitSpeechRecognition();
        recognition.continuous = false;
        recognition.interimResults = false;
        recognition.lang = currentLanguage === 'fr' ? 'fr-FR' : 'en-US';

        recognition.onstart = function() {
            isRecording = true;
            updateMicButton(true);
            document.getElementById('speech-status').style.display = 'block';
        };

        recognition.onend = function() {
            isRecording = false;
            updateMicButton(false);
            document.getElementById('speech-status').style.display = 'none';
        };

        recognition.onresult = function(event) {
            const transcript = event.results[0][0].transcript;
            document.getElementById('user-input').value = transcript;
            sendMessage();
        };

        recognition.onerror = function(event) {
            console.error('Speech recognition error:', event.error);
            isRecording = false;
            updateMicButton(false);
            document.getElementById('speech-status').style.display = 'none';
        };
    } else {
        alert('Speech recognition is not supported in this browser. Please use Chrome.');
    }
}

function toggleSpeech() {
    if (!recognition) {
        initSpeechRecognition();
    }

    if (recognition) {
        if (!isRecording) {
            recognition.start();
        } else {
            recognition.stop();
        }
    }
}

function updateMicButton(recording) {
    const micButton = document.getElementById('mic-button');
    if (recording) {
        micButton.classList.add('recording');
    } else {
        micButton.classList.remove('recording');
    }
}