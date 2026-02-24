📘 EDUbot – AI Learning Assistant (Gemini 2.5 + ML Intent Model)

EDUbot is a desktop-based AI chatbot built using Tkinter + CustomTkinter, powered by:

Google Gemini 2.5 Flash for natural and academic responses

A neural network (TensorFlow) for intent detection

Rule-based engine for study materials & class scheduling

JSON NLP dataset for greetings, exam help, motivation, etc.

This project combines AI + ML + NLP to create a smart educational assistant.

🚀 Features

✔️ AI-powered responses using Gemini 2.5 Flash
✔️ Study material generation (videos, PDFs, articles)
✔️ Neural-network based intent classification
✔️ Class scheduling (dummy auto-generated schedule)
✔️ Friendly GUI using CustomTkinter
✔️ JSON-based conversational patterns
✔️ Works fully offline except Gemini API calls

📂 Project Structure
📁 CHAT-bot/
│── edu_chatbot_gui.py        # Main application
│── intents.json              # NLP training data
│── chatbot_model.h5          # Trained ML intent model
│── classes.pkl               # Encoded classes for model
│── words.pkl                 # Vocabulary
│── requirements.txt          # Python dependencies
│── model_training.ipynb      # Training notebook
│── .gitignore
│── README.md
🔧 Installation & Setup
1️⃣ Clone the repository
git clone https://github.com/bikram3369/CHAT-bot.git
cd CHAT-bot
2️⃣ Create a virtual environment
python -m venv .venv
Activate it:
Windows PowerShell
.\.venv\Scripts\Activate.ps1
Windows CMD
.\.venv\Scripts\activate.bat
Linux / macOS
source .venv/bin/activate
3️⃣ Install all dependencies
pip install -r requirements.txt
4️⃣ Add your Gemini API Key

Create a file named:

.env

Put your key inside:

GEMINI_API_KEY=your_api_key_here

Or directly paste into code (not recommended for production).

5️⃣ Run the application
python edu_chatbot_gui.py
🧠 How It Works
🔹 1. ML Intent Detection

Model (chatbot_model.h5) predicts user intent

Uses word bag vectors from words.pkl

Uses classes from classes.pkl

🔹 2. Rule Engine

Handles:

Study materials

Class scheduling

Assignment checks

🔹 3. Gemini AI

Used when:

Intent is unknown

User asks complex academic queries

User wants study materials

User initiates class scheduling

📚 Training Your Own Model

Inside model_training.ipynb you can:

Add new intents

Retrain model

Generate updated .pkl + .h5 files

🛠 Technologies Used

Python 3.10+

TensorFlow

NLTK

NumPy

CustomTkinter

Google Gemini API

🤝 Contribution

Contributions are welcome!

Fork repo

Create feature branch

Commit changes

Open a Pull Request

⭐ Show Support

If you like this project, give it a ⭐ on GitHub
It motivates further updates & improvements!
