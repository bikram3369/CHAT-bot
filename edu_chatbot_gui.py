# import customtkinter as ctk
# import tkinter as tk
# import json
# import pickle
# import numpy as np
# import random
# import nltk
# from nltk.stem import WordNetLemmatizer
# from tensorflow.keras.models import load_model

# # -----------------------------------------
# # ⭐ GEMINI 2.5 FLASH (CONFIRMED WORKING)
# # -----------------------------------------
# import google.generativeai as genai

# API_KEY = "your api key here"
# genai.configure(api_key=API_KEY)

# gemini = genai.GenerativeModel("models/gemini-2.5-flash")


# def gemini_search(query):
#     """Search online using Gemini"""
#     try:
#         prompt = f"""
#         You are EDUbot, an AI education assistant.
#         Provide BEST study materials for: {query}

#         Include:
#         ### Summary (3–4 lines)

#         ### Videos
#         - 5 YouTube links

#         ### PDFs
#         - 3 links

#         ### Articles
#         - 3 links
#         """

#         response = gemini.generate_content(prompt)
#         return response.text
#     except Exception as e:
#         return f"⚠ Gemini Error: {str(e)}"


# # -----------------------------------------
# # ⭐ LOAD TRAINED MODEL + NLP DATA
# # -----------------------------------------
# lemmatizer = WordNetLemmatizer()

# model = load_model("chatbot_model.h5")
# intents = json.loads(open("intents.json").read())
# words = pickle.load(open("words.pkl", "rb"))
# classes = pickle.load(open("classes.pkl", "rb"))


# def clean(sentence):
#     tokens = nltk.word_tokenize(sentence)
#     return [lemmatizer.lemmatize(w.lower()) for w in tokens]


# def bow(sentence):
#     sentence_words = clean(sentence)
#     return np.array([1 if w in sentence_words else 0 for w in words])


# def predict_class(sentence):
#     bow_vec = bow(sentence)
#     preds = model.predict(np.array([bow_vec]))[0]
#     threshold = 0.30

#     results = [[i, p] for i, p in enumerate(preds) if p > threshold]
#     results.sort(key=lambda x: x[1], reverse=True)

#     return [{"intent": classes[r[0]], "prob": str(r[1])} for r in results]


# def ml_answer(ints):
#     if len(ints) == 0:
#         return None

#     tag = ints[0]["intent"]
#     for intent in intents["intents"]:
#         if intent["tag"] == tag:
#             return random.choice(intent["responses"])

#     return None


# # -----------------------------------------
# # ⭐ RULE ENGINE
# # -----------------------------------------
# def rule_engine(msg):
#     text = msg.lower().strip()

#     subjects = {
#         "math": "mathematics",
#         "maths": "mathematics",
#         "physics": "physics",
#         "chemistry": "chemistry",
#         "bio": "biology",
#         "biology": "biology",
#         "computer": "computer science",
#         "cs": "computer science",
#         "english": "english",
#         "history": "history",
#         "geography": "geography",
#     }

#     for key, subject in subjects.items():
#         if key in text:
#             return gemini_search(f"{subject} study materials")

#     if "assignment" in text:
#         return "Sure! Paste your assignment text here."

#     if "schedule" in text or "class" in text:
#         return "Which subject should I schedule a class for?"

#     if "exam" in text:
#         return gemini_search(f"exam preparation for {msg}")

#     if "course" in text:
#         return "Tell me the course name."

#     return None


# # -----------------------------------------
# # ⭐ CUSTOMTKINTER MODERN GUI
# # -----------------------------------------
# ctk.set_appearance_mode("dark")
# ctk.set_default_color_theme("blue")


# class EduBot(ctk.CTk):
#     def __init__(self):
#         super().__init__()

#         self.title("EDUbot – Gemini 2.5 AI Learning Assistant")
#         self.geometry("900x900")

#         # CHAT WINDOW (VERTICAL AUTO WRAP, NO CUT)
#         self.chat = ctk.CTkScrollableFrame(self, width=860, height=750)
#         self.chat.pack(pady=20, padx=20)

#         # INPUT FIELD
#         self.entry = ctk.CTkEntry(self, width=650, placeholder_text="Type your message…")
#         self.entry.pack(side="left", padx=10, pady=10)
#         self.entry.bind("<Return>", self.send_message)

#         # SEND BUTTON
#         self.send_btn = ctk.CTkButton(self, text="Send", width=150, command=self.send_message)
#         self.send_btn.pack(side="right", padx=10)

#         self.bot("Hello! 👋 I'm EDUbot (Gemini 2.5). Ask me anything!")

#     # ---------------- Display Messages ----------------
#     def user(self, msg):
#         lbl = ctk.CTkLabel(self.chat, text=f"You: {msg}", anchor="w", wraplength=820)
#         lbl.pack(fill="x", pady=5)

#     def bot(self, msg):
#         lbl = ctk.CTkLabel(self.chat, text=f"EDUbot: {msg}", anchor="w",
#                            text_color="#00ffcc", wraplength=820, justify="left")
#         lbl.pack(fill="x", pady=5)

#     # ---------------- SEND LOGIC ----------------
#     def send_message(self, event=None):
#         message = self.entry.get().strip()
#         if message == "":
#             return

#         self.entry.delete(0, tk.END)
#         self.user(message)

#         # 3️⃣ GEMINI SEARCH (FALLBACK)
#         result = gemini_search(message + " study materials")
#         self.bot(result)

#         # 1️⃣ RULE ENGINE
#         rule = rule_engine(message)
#         if rule:
#             self.bot(rule)
#             return

#         # 2️⃣ ML PREDICTION
#         intents_pred = predict_class(message)
#         ml_ans = ml_answer(intents_pred)
#         if ml_ans:
#             self.bot(ml_ans)
#             return



# # RUN APP
# app = EduBot()
# app.mainloop()




# import customtkinter as ctk
# import tkinter as tk
# import json
# import pickle
# import numpy as np
# import random
# import nltk
# from nltk.stem import WordNetLemmatizer
# from tensorflow.keras.models import load_model

# # -----------------------------------------
# # ⭐ GEMINI 2.5 FLASH WORKING
# # -----------------------------------------
# import google.generativeai as genai

# API_KEY = "your api key here"
# genai.configure(api_key=API_KEY)

# gemini = genai.GenerativeModel("models/gemini-2.5-flash")


# def gemini_search(query):
#     """Study Material Search"""
#     try:
#         prompt = f"""
# You are EDUbot, an AI assistant.
# Provide BEST study materials for: {query}

# ### Summary (3–4 lines)

# ### Videos
# - 5 YouTube links

# ### PDFs
# - 3 links

# ### Articles
# - 3 links
# """
#         response = gemini.generate_content(prompt)
#         return response.text
#     except Exception as e:
#         return f"⚠ Gemini Error: {str(e)}"


# def gemini_answer(query):
#     """Normal Gemini Answer"""
#     try:
#         response = gemini.generate_content(query)
#         return response.text
#     except Exception as e:
#         return f"⚠ Gemini Error: {str(e)}"


# # -----------------------------------------
# # ⭐ ML MODEL (json + neural network)
# # -----------------------------------------
# lemmatizer = WordNetLemmatizer()

# model = load_model("chatbot_model.h5")
# intents = json.loads(open("intents.json").read())
# words = pickle.load(open("words.pkl", "rb"))
# classes = pickle.load(open("classes.pkl", "rb"))


# def clean(sentence):
#     return [lemmatizer.lemmatize(w.lower()) for w in nltk.word_tokenize(sentence)]


# def bow(sentence):
#     sw = clean(sentence)
#     return np.array([1 if w in sw else 0 for w in words])


# def predict_class(sentence):
#     bow_vec = bow(sentence)
#     preds = model.predict(np.array([bow_vec]))[0]

#     threshold = 0.70  # high confidence
#     results = [[i, p] for i, p in enumerate(preds) if p > threshold]
#     results.sort(key=lambda x: x[1], reverse=True)

#     return [{"intent": classes[r[0]]} for r in results]


# def ml_answer(intents_list):
#     if not intents_list:
#         return None

#     tag = intents_list[0]["intent"]
#     for intent in intents["intents"]:
#         if intent["tag"] == tag:
#             return random.choice(intent["responses"])
#     return None


# # -----------------------------------------
# # ⭐ RULE ENGINE (study materials & scheduling)
# # -----------------------------------------
# def rule_engine(msg):
#     msg = msg.lower()

#     subjects = {
#         "math": "mathematics",
#         "maths": "mathematics",
#         "physics": "physics",
#         "chemistry": "chemistry",
#         "biology": "biology",
#         "bio": "biology",
#         "computer": "computer science",
#         "cs": "computer science",
#         "english": "english",
#         "history": "history",
#         "geography": "geography",
#     }

#     # Study materials
#     for key, subject in subjects.items():
#         if key in msg:
#             return gemini_search(f"{subject} study materials")

#     # Scheduling classes
#     if "schedule" in msg or "class" in msg or "book" in msg:
#         return gemini_answer("Please provide subject and preferred class timing.")

#     return None  # no rule matched


# # -----------------------------------------
# # ⭐ GUI
# # -----------------------------------------
# ctk.set_appearance_mode("dark")
# ctk.set_default_color_theme("blue")


# class EduBot(ctk.CTk):
#     def __init__(self):
#         super().__init__()

#         self.title("EDUbot – Gemini 2.5 AI Learning Assistant")
#         self.geometry("900x900")

#         self.chat = ctk.CTkScrollableFrame(self, width=860, height=750)
#         self.chat.pack(pady=20, padx=20)

#         self.entry = ctk.CTkEntry(self, width=650, placeholder_text="Type your message…")
#         self.entry.pack(side="left", padx=10, pady=10)
#         self.entry.bind("<Return>", self.send_message)

#         self.send_btn = ctk.CTkButton(self, text="Send", command=self.send_message)
#         self.send_btn.pack(side="right", padx=10)

#         self.bot("Hello! 👋 I'm EDUbot. Ask me anything!")

#     def user(self, msg):
#         ctk.CTkLabel(self.chat, text=f"You: {msg}", wraplength=820).pack(fill="x", pady=5)

#     def bot(self, msg):
#         ctk.CTkLabel(
#             self.chat, text=f"EDUbot: {msg}", wraplength=820, text_color="#00ffcc"
#         ).pack(fill="x", pady=5)

#     def send_message(self, event=None):
#         msg = self.entry.get().strip()
#         if msg == "":
#             return

#         self.entry.delete(0, tk.END)
#         self.user(msg)

#         # 1️⃣ RULE ENGINE
#         rule = rule_engine(msg)
#         if rule:
#             self.bot(rule)
#             return

#         # 2️⃣ ML INTENT (hi, hello, what is JLPT)
#         ml = ml_answer(predict_class(msg))
#         if ml:
#             self.bot(ml)
#             return

#         # 3️⃣ GEMINI GENERAL ANSWER (fallback)
#         self.bot(gemini_answer(msg))


# app = EduBot()
# app.mainloop()

import customtkinter as ctk
import tkinter as tk
import json
import pickle
import numpy as np
import random
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

# -----------------------------------------
# ⭐ GEMINI 2.5 FLASH (WORKING)
# -----------------------------------------
import google.generativeai as genai

API_KEY = "your  api key here"
genai.configure(api_key=API_KEY)

gemini = genai.GenerativeModel("models/gemini-2.5-flash")


def gemini_search(query):
    """Study Material Search"""
    try:
        prompt = f"""
You are EDUbot, an AI assistant.
Provide BEST study materials for: {query}

### Summary (3–4 lines)

### Videos
- 5 YouTube links

### PDFs
- 3 links

### Articles
- 3 links
"""
        response = gemini.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"⚠ Gemini Error: {str(e)}"


def gemini_answer(query):
    """General conversation"""
    try:
        response = gemini.generate_content(query)
        return response.text
    except Exception as e:
        return f"⚠ Gemini Error: {str(e)}"


# -----------------------------------------
# ⭐ ML MODEL (hi, hello, what is JLPT etc.)
# -----------------------------------------
lemmatizer = WordNetLemmatizer()

model = load_model("chatbot_model.h5")
intents = json.loads(open("intents.json").read())
words = pickle.load(open("words.pkl", "rb"))
classes = pickle.load(open("classes.pkl", "rb"))


def clean(sentence):
    return [lemmatizer.lemmatize(w.lower()) for w in nltk.word_tokenize(sentence)]


def bow(sentence):
    sw = clean(sentence)
    return np.array([1 if w in sw else 0 for w in words])


def predict_class(sentence):
    bow_vec = bow(sentence)
    preds = model.predict(np.array([bow_vec]))[0]

    threshold = 0.70
    results = [[i, p] for i, p in enumerate(preds) if p > threshold]
    results.sort(key=lambda x: x[1], reverse=True)

    return [{"intent": classes[r[0]]} for r in results]


def ml_answer(intents_list):
    if not intents_list:
        return None

    tag = intents_list[0]["intent"]
    for intent in intents["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])
    return None


# -----------------------------------------
# ⭐ AUTO CLASS SCHEDULER
# -----------------------------------------
def schedule_class_auto(message):
    subjects = {
        "math": "Mathematics",
        "maths": "Mathematics",
        "physics": "Physics",
        "chemistry": "Chemistry",
        "biology": "Biology",
        "bio": "Biology",
        "computer": "Computer Science",
        "cs": "Computer Science",
    }

    subject_found = None
    msg_lower = message.lower()

    for key, subj in subjects.items():
        if key in msg_lower:
            subject_found = subj
            break

    if subject_found is None:
        return "Sure! Please tell me which subject you want the class for."

    teachers = [
        "Mr. Sharma", "Ms. Gupta", "Dr. Verma",
        "Mr. Iyer", "Mrs. Banerjee", "Dr. Sen"
    ]

    timings = [
        "10:00 AM", "11:30 AM", "1:00 PM",
        "3:00 PM", "4:30 PM", "6:00 PM"
    ]

    teacher = random.choice(teachers)
    timing = random.choice(timings)

    return (
        f"📚 Your {subject_found} class has been scheduled!\n"
        f"👨‍🏫 Teacher: {teacher}\n"
        f"⏰ Time: {timing}\n"
        f"Let me know if you want to reschedule!"
    )


# -----------------------------------------
# ⭐ RULE ENGINE (study materials only)
# -----------------------------------------
def rule_engine(msg):
    msg = msg.lower()

    subjects = {
        "math": "mathematics",
        "maths": "mathematics",
        "physics": "physics",
        "chemistry": "chemistry",
        "biology": "biology",
        "bio": "biology",
        "computer": "computer science",
        "cs": "computer science",
        "english": "english",
        "history": "history",
        "geography": "geography",
    }

    for key, subject in subjects.items():
        if key in msg:
            return gemini_search(f"{subject} study materials")

    return None


# -----------------------------------------
# ⭐ GUI
# -----------------------------------------
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class EduBot(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("EDUbot – Gemini 2.5 Learning Assistant")
        self.geometry("900x900")

        self.chat = ctk.CTkScrollableFrame(self, width=860, height=750)
        self.chat.pack(pady=20, padx=20)

        self.entry = ctk.CTkEntry(self, width=650, placeholder_text="Type your message…")
        self.entry.pack(side="left", padx=10, pady=10)
        self.entry.bind("<Return>", self.send_message)

        self.send_btn = ctk.CTkButton(self, text="Send", command=self.send_message)
        self.send_btn.pack(side="right", padx=10)

        self.bot("Hello! 👋 I'm EDUbot. Ask me anything!")

    def user(self, msg):
        ctk.CTkLabel(self.chat, text=f"You: {msg}", wraplength=820).pack(fill="x", pady=5)

    def bot(self, msg):
        ctk.CTkLabel(
            self.chat, text=f"EDUbot: {msg}", wraplength=820, text_color="#00ffcc"
        ).pack(fill="x", pady=5)

    def send_message(self, event=None):
        msg = self.entry.get().strip()
        if msg == "":
            return

        self.entry.delete(0, tk.END)
        self.user(msg)

        # 0️⃣ AUTO CLASS SCHEDULING
        schedule_keywords = ["schedule", "book", "class", "fix", "lecture"]
        if any(word in msg.lower() for word in schedule_keywords):
            reply = schedule_class_auto(msg)
            self.bot(reply)
            return

        # 1️⃣ STUDY MATERIAL RULE ENGINE
        rule = rule_engine(msg)
        if rule:
            self.bot(rule)
            return

        # 2️⃣ ML INTENT (hi, hello, what is JLPT)
        ml = ml_answer(predict_class(msg))
        if ml:
            self.bot(ml)
            return

        # 3️⃣ GEMINI GENERAL ANSWER (fallback)
        self.bot(gemini_answer(msg))


app = EduBot()
app.mainloop()