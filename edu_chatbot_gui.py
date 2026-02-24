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
# # ⭐ GEMINI 2.5 FLASH (WORKING)
# # -----------------------------------------
# import google.generativeai as genai

# API_KEY = "your  api key here"
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
#     """General conversation"""
#     try:
#         response = gemini.generate_content(query)
#         return response.text
#     except Exception as e:
#         return f"⚠ Gemini Error: {str(e)}"


# # -----------------------------------------
# # ⭐ ML MODEL (hi, hello, what is JLPT etc.)
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

#     threshold = 0.70
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
# # ⭐ AUTO CLASS SCHEDULER
# # -----------------------------------------
# def schedule_class_auto(message):
#     subjects = {
#         "math": "Mathematics",
#         "maths": "Mathematics",
#         "physics": "Physics",
#         "chemistry": "Chemistry",
#         "biology": "Biology",
#         "bio": "Biology",
#         "computer": "Computer Science",
#         "cs": "Computer Science",
#     }

#     subject_found = None
#     msg_lower = message.lower()

#     for key, subj in subjects.items():
#         if key in msg_lower:
#             subject_found = subj
#             break

#     if subject_found is None:
#         return "Sure! Please tell me which subject you want the class for."

#     teachers = [
#         "Mr. Sharma", "Ms. Gupta", "Dr. Verma",
#         "Mr. Iyer", "Mrs. Banerjee", "Dr. Sen"
#     ]

#     timings = [
#         "10:00 AM", "11:30 AM", "1:00 PM",
#         "3:00 PM", "4:30 PM", "6:00 PM"
#     ]

#     teacher = random.choice(teachers)
#     timing = random.choice(timings)

#     return (
#         f"📚 Your {subject_found} class has been scheduled!\n"
#         f"👨‍🏫 Teacher: {teacher}\n"
#         f"⏰ Time: {timing}\n"
#         f"Let me know if you want to reschedule!"
#     )


# # -----------------------------------------
# # ⭐ RULE ENGINE (study materials only)
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

#     for key, subject in subjects.items():
#         if key in msg:
#             return gemini_search(f"{subject} study materials")

#     return None


# # -----------------------------------------
# # ⭐ GUI
# # -----------------------------------------
# ctk.set_appearance_mode("dark")
# ctk.set_default_color_theme("blue")


# class EduBot(ctk.CTk):
#     def __init__(self):
#         super().__init__()

#         self.title("EDUbot – Gemini 2.5 Learning Assistant")
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

#         # 0️⃣ AUTO CLASS SCHEDULING
#         schedule_keywords = ["schedule", "book", "class", "fix", "lecture"]
#         if any(word in msg.lower() for word in schedule_keywords):
#             reply = schedule_class_auto(msg)
#             self.bot(reply)
#             return

#         # 1️⃣ STUDY MATERIAL RULE ENGINE
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

# ---------------- GEMINI SETUP ----------------
import google.generativeai as genai

API_KEY = "your api key here"
genai.configure(api_key=API_KEY)

gemini = genai.GenerativeModel("models/gemini-2.5-flash")


def gemini_search(query):
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
    try:
        response = gemini.generate_content(query)
        return response.text
    except Exception as e:
        return f"⚠ Gemini Error: {str(e)}"


# ---------------- ML MODEL ----------------
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


# ---------------- AUTO SCHEDULING ----------------
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

    teachers = ["Mr. Sharma", "Ms. Gupta", "Dr. Verma",
                "Mr. Iyer", "Mrs. Banerjee", "Dr. Sen"]

    timings = ["10:00 AM", "11:30 AM", "1:00 PM",
               "3:00 PM", "4:30 PM", "6:00 PM"]

    teacher = random.choice(teachers)
    timing = random.choice(timings)

    return (
        f"📚 Your {subject_found} class has been scheduled!\n"
        f"👨‍🏫 Teacher: {teacher}\n"
        f"⏰ Time: {timing}\n"
        f"Let me know if you want to reschedule!"
    )


# ---------------- RULE ENGINE ----------------
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


# ---------------- CHATGPT-STYLE GUI ----------------
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class EduBot(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("EDUbot – Gemini 2.5 ChatGPT Style")
        self.geometry("900x900")

        # Main Chat Frame
        self.chat = ctk.CTkScrollableFrame(self, width=860, height=750, fg_color="#0f0f0f")
        self.chat.pack(pady=20, padx=20)

        # Input box
        bottom_frame = ctk.CTkFrame(self, fg_color="#0f0f0f")
        bottom_frame.pack(pady=10, padx=10, fill="x")

        self.entry = ctk.CTkEntry(bottom_frame, width=700, placeholder_text="Message EDUbot…")
        self.entry.pack(side="left", padx=10, pady=10)
        self.entry.bind("<Return>", self.send_message)

        self.send_btn = ctk.CTkButton(bottom_frame, text="Send", command=self.send_message)
        self.send_btn.pack(side="right", padx=10)

        self.bot("Hello! 👋 How can I help you today?")

    # USER bubble (Blue, Right)
    def user(self, msg):
        bubble = ctk.CTkLabel(
            self.chat,
            text=msg,
            fg_color="#0055ff",
            text_color="white",
            corner_radius=18,
            justify="left",
            wraplength=700,
            anchor="e",
            padx=15,
            pady=10
        )
        bubble.pack(anchor="e", pady=5, padx=10)

    # BOT bubble (Grey, Left)
    def bot(self, msg):
        bubble = ctk.CTkLabel(
            self.chat,
            text=msg,
            fg_color="#1e1e1e",
            text_color="#00ffcc",
            corner_radius=18,
            justify="left",
            wraplength=700,
            anchor="w",
            padx=15,
            pady=10
        )
        bubble.pack(anchor="w", pady=5, padx=10)

    def send_message(self, event=None):
        msg = self.entry.get().strip()
        if not msg:
            return

        self.entry.delete(0, tk.END)
        self.user(msg)

        # Scheduling
        schedule_keywords = ["schedule", "book", "class", "fix", "lecture"]
        if any(word in msg.lower() for word in schedule_keywords):
            self.bot(schedule_class_auto(msg))
            return

        # Study materials
        rule = rule_engine(msg)
        if rule:
            self.bot(rule)
            return

        # ML intents
        ml = ml_answer(predict_class(msg))
        if ml:
            self.bot(ml)
            return

        # Gemini fallback
        self.bot(gemini_answer(msg))


app = EduBot()
app.mainloop()