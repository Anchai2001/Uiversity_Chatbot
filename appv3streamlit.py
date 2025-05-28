# -*- coding: utf-8 -*-
"""
Streamlit app for University Schedule Chatbot
"""

import streamlit as st
import spacy
import pandas as pd
from datetime import datetime, time
import random
from spacy.matcher import PhraseMatcher
import pickle
from config import LANGUAGES, UNIVERSITY_TERMS
from data_loader2 import load_schedule_data
from fuzzywuzzy import fuzz, process

class UniversityChatbot:
    def __init__(self, excel_path):
        try:
            self.df = load_schedule_data(excel_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Excel file {excel_path} not found.")
        self.normalized_courses = self._preprocess_courses()
        self.normalized_teachers = self._preprocess_teachers()
        self.normalized_teachers_no_titles = self._preprocess_teachers_no_titles()
        try:
            self.nlp_en = self._setup_nlp('en')
            self.nlp_tr = self._setup_nlp('tr')
        except OSError as e:
            raise OSError(f"Failed to load SpaCy model: {e}")
        try:
            self.model = pickle.load(open('intent_classifier.pkl', 'rb'))
            self.vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
        except FileNotFoundError:
            raise FileNotFoundError("ML models not found.")
        self.context = {
            'history': [],
            'last_course': None,
            'last_teacher': None,
            'last_day': None
        }

    def _preprocess_courses(self):
        courses = self.df['Course'].unique()
        return {self._normalize_text(course): course for course in courses if isinstance(course, str)}

    def _preprocess_teachers(self):
        teachers = self.df['Teacher'].unique()
        return {self._normalize_text(teacher): teacher for teacher in teachers if isinstance(teacher, str) and teacher != '-'}

    def _preprocess_teachers_no_titles(self):
        teachers = self.df['Teacher'].unique()
        title_prefixes = ['Doç. Dr.', 'Öğr. Gör. Dr.', 'Öğr. Gör.', 'Prof. Dr.', 'Dr.']
        no_title_map = {}
        for teacher in teachers:
            if isinstance(teacher, str) and teacher != '-':
                no_title = teacher
                for prefix in title_prefixes:
                    if no_title.startswith(prefix):
                        no_title = no_title[len(prefix):].strip()
                no_title_map[self._normalize_text(no_title)] = teacher
        return no_title_map

    def _normalize_text(self, text):
        return text.lower().strip()

    def _format_time(self, time_obj):
        if isinstance(time_obj, time):
            return time_obj.strftime('%H:%M')
        return str(time_obj) if time_obj else "time not set"

    def _setup_nlp(self, lang):
        try:
            nlp = spacy.load(LANGUAGES[lang]['model'])
        except OSError:
            raise ValueError(f"Model {LANGUAGES[lang]['model']} not found.")
        if "entity_ruler" not in nlp.pipe_names:
            nlp.add_pipe("entity_ruler", before="ner")
        ruler = nlp.get_pipe("entity_ruler")
        patterns = []
        for course in self.normalized_courses.values():
            patterns.append({"label": "COURSE", "pattern": course})
        for teacher in self.normalized_teachers.values():
            patterns.append({"label": "TEACHER", "pattern": teacher})
        ruler.add_patterns(patterns)
        matcher = PhraseMatcher(nlp.vocab)
        for label, terms in LANGUAGES[lang]['keywords'].items():
            matcher.add(label.upper(), [nlp.make_doc(text) for text in terms])
        if lang == 'en':
            self.matcher_en = matcher
        else:
            self.matcher_tr = matcher
        return nlp

    def detect_language(self, text):
        tr_words = ['ders', 'hoca', 'vize', 'final']
        return 'tr' if any(word in text.lower() for word in tr_words) else 'en'

    def predict_intent(self, text):
        processed = self._normalize_text(text)
        vec = self.vectorizer.transform([processed])
        return self.model.predict(vec)[0]

    def fuzzy_match_entity(self, text, candidates, threshold=70):
        if not text:
            return None
        normalized_text = self._normalize_text(text)
        match, score = process.extractOne(normalized_text, candidates.keys(), scorer=fuzz.token_sort_ratio)
        if score >= threshold:
            return candidates[match]
        return None

    def _update_context(self, query, intent, entities):
        interaction = {
            'query': query,
            'intent': intent,
            'entities': entities.copy(),
            'timestamp': datetime.now()
        }
        self.context['history'].append(interaction)
        if len(self.context['history']) > 5:
            self.context['history'] = self.context['history'][-5:]
        self.context['last_course'] = entities.get('course', self.context['last_course'])
        self.context['last_teacher'] = entities.get('teacher', self.context['last_teacher'])
        self.context['last_day'] = entities.get('day', self.context['last_day'])

    def extract_entities(self, text):
        normalized = self._normalize_text(text)
        lang = self.detect_language(text)
        nlp = self.nlp_tr if lang == 'tr' else self.nlp_en
        matcher = self.matcher_tr if lang == 'tr' else self.matcher_en
        doc = nlp(text)

        entities = {
            'course': None,
            'teacher': None,
            'exam_type': None,
            'day': None,
            'time': None,
            'building': None
        }

        pronouns = {
            'en': ['he', 'she', 'her', 'him', 'it'],
            'tr': ['o', 'onun', 'ona']
        }
        is_followup = any(pronoun in normalized for pronoun in pronouns[lang]) or 'about' in normalized or 'hakkında' in normalized

        for norm_name, original in self.normalized_courses.items():
            if norm_name in normalized:
                entities['course'] = original
                break
        for norm_name, original in self.normalized_teachers.items():
            if norm_name in normalized:
                entities['teacher'] = original
                break

        if not entities['course']:
            entities['course'] = self.fuzzy_match_entity(normalized, self.normalized_courses)
        if not entities['teacher']:
            entities['teacher'] = self.fuzzy_match_entity(normalized, self.normalized_teachers)
            if not entities['teacher']:
                entities['teacher'] = self.fuzzy_match_entity(normalized, self.normalized_teachers_no_titles)

        for ent in doc.ents:
            if ent.label_ == "COURSE":
                teacher_match = self.fuzzy_match_entity(ent.text, self.normalized_teachers_no_titles)
                if teacher_match:
                    entities['teacher'] = teacher_match
                    entities['course'] = None
                else:
                    entities['course'] = ent.text
            elif ent.label_ == "TEACHER":
                entities['teacher'] = ent.text
            elif ent.label_ == "DATE":
                entities['day'] = ent.text
            elif ent.label_ == "TIME":
                entities['time'] = ent.text
            elif ent.label_ == "PERSON" and not entities['teacher']:
                entities['teacher'] = self.fuzzy_match_entity(ent.text, self.normalized_teachers_no_titles)

        if is_followup or not any([entities['course'], entities['teacher']]):
            for interaction in reversed(self.context['history']):
                if not entities['teacher'] and interaction['entities'].get('teacher'):
                    entities['teacher'] = interaction['entities']['teacher']
                    break
                if not entities['course'] and interaction['entities'].get('course'):
                    entities['course'] = interaction['entities']['course']
                    break

        turkish_days = {
            "pazartesi": "Pazartesi",
            "salı": "Salı",
            "çarşamba": "Çarşamba",
            "perşembe": "Perşembe",
            "cuma": "Cuma",
            "cumartesi": "Cumartesi",
            "pazar": "Pazar"
        }
        for token in doc:
            lower = token.text.lower()
            if lower in turkish_days:
                entities["day"] = turkish_days[lower]

        keyword_to_entity = {
            'course': 'course',
            'exam': 'exam_type',
            'teacher': 'teacher',
        }
        matches = matcher(doc)
        for match_id, start, end in matches:
            label = nlp.vocab.strings[match_id].lower()
            if label in keyword_to_entity:
                key = keyword_to_entity[label]
                if key == 'course' and entities['teacher']:
                    continue
                entities[key] = doc[start:end].text

        for token in doc:
            if token.text in UNIVERSITY_TERMS['buildings']:
                entities['building'] = token.text

        exam_keywords = {
            'en': ['midterm', 'final', 'makeup'],
            'tr': ['vize', 'final', 'bütünleme']
        }
        for token in doc:
            if token.lemma_.lower() in exam_keywords[lang]:
                entities['exam_type'] = token.lemma_.lower()

        return entities

    def generate_response(self, text, entities):
        intent = self.predict_intent(text)
        self._update_context(text, intent, entities)
        if intent == "greeting":
            return random.choice(["Merhaba! 📚 How can I assist you today?", "Hi! Ready to help with your schedule. 😊"])
        elif intent == "goodbye":
            return random.choice(["Görüşürüz! 👋", "Bye! Take care! 😊"])
        elif intent == "thanks":
            return random.choice(["Rica ederim! 😊", "You're welcome!"])
        elif intent == "class_schedule":
            course = entities.get("course") or self.context["last_course"]
            day = entities.get("day") or self.context["last_day"]
            if course:
                df = self.df[(self.df["Course"] == course) & (self.df["Exam Type"] == "Lecture")]
                if day:
                    df = df[df["Day"] == day]
                if not df.empty:
                    response = f"📚 Schedule for {course}:\n"
                    for _, row in df.iterrows():
                        if pd.notna(row["Time"]) and row["Day"]:
                            response += f"- {row['Day']} at {self._format_time(row['Time'])} in room {row['Room']}\n"
                    return response.strip()
                return f"No schedule found for {course}."
            return "Which course's schedule would you like to know?"
        elif intent == "exam_info":
            course = entities.get("course") or self.context["last_course"]
            exam_type = entities.get("exam_type")
            if course:
                df = self.df[(self.df["Course"] == course) & (self.df["Exam Type"].isin(["Midterm", "Final", "Makeup"]))]
                if exam_type:
                    df = df[df["Exam Type"].str.lower() == exam_type.lower()]
                if not df.empty:
                    response = f"📝 Exam info for {course}:\n"
                    for _, row in df.iterrows():
                        if pd.notna(row["Exam Date"]):
                            date = row["Exam Date"].strftime("%d.%m.%Y")
                            time = self._format_time(row["Exam Time"])
                            response += f"- {row['Exam Type']} on {date} at {time}\n"
                    return response.strip()
                return f"No exam info found for {course}."
            return "Which course's exam info would you like?"
        elif intent == "teacher_info":
            teacher = entities.get("teacher") or self.context["last_teacher"]
            if teacher:
                df = self.df[(self.df["Teacher"] == teacher) & (self.df["Exam Type"] == "Lecture")]
                if not df.empty:
                    courses = df["Course"].unique()
                    return f"👨‍🏫 {teacher} teaches: {', '.join(courses)}"
                return f"No courses found for {teacher}."
            return "Which teacher would you like info about?"
        elif intent == "room_info":
            course = entities.get("course") or self.context["last_course"]
            day = entities.get("day") or self.context["last_day"]
            if course:
                df = self.df[(self.df["Course"] == course) & (self.df["Exam Type"] == "Lecture")]
                if day:
                    df = df[df["Day"] == day]
                if not df.empty:
                    response = f"🏫 Rooms for {course}:\n"
                    for _, row in df.iterrows():
                        if row["Room"] and pd.notna(row["Time"]):
                            response += f"- {row['Day']} at {self._format_time(row['Time'])} in room {row['Room']}\n"
                    return response.strip()
                return f"No room info found for {course}."
            return "Which course's room info would you like?"
        elif intent == "daily_schedule":
            day = entities.get("day") or self.context["last_day"]
            if day:
                df = self.df[(self.df["Day"] == day) & (self.df["Exam Type"] == "Lecture")]
                if not df.empty:
                    response = f"📅 Schedule for {day}:\n"
                    for _, row in df.iterrows():
                        if pd.notna(row["Time"]):
                            response += f"- {row['Course']} at {self._format_time(row['Time'])} in room {row['Room']} (Teacher: {row['Teacher']})\n"
                    return response.strip()
                return f"No classes found for {day}."
            return "Which day's schedule would you like?"
        elif intent == "schedule_conflict":
            df = self.df[self.df["Exam Type"] == "Lecture"]
            conflicts = []
            for day in df["Day"].unique():
                day_df = df[df["Day"] == day]
                for time in day_df["Time"].unique():
                    if pd.notna(time):
                        time_df = day_df[day_df["Time"] == time]
                        if len(time_df) > 1:
                            conflicts.append((day, time, time_df["Course"].tolist()))
            if conflicts:
                response = "⚠️ Schedule conflicts:\n"
                for day, time, courses in conflicts:
                    response += f"- {day} at {self._format_time(time)}: {', '.join(courses)}\n"
                return response.strip()
            return "No schedule conflicts found."
        elif intent == "teacher_schedule":
            teacher = entities.get("teacher") or self.context["last_teacher"]
            if teacher:
                df = self.df[(self.df["Teacher"] == teacher) & (self.df["Exam Type"] == "Lecture")]
                if not df.empty:
                    response = f"👨‍🏫 {teacher}'s teaching schedule:\n"
                    for _, row in df.iterrows():
                        if pd.notna(row["Time"]) and row["Day"]:
                            response += f"- {row['Course']} on {row['Day']} at {self._format_time(row['Time'])} in room {row['Room']}\n"
                    return response.strip()
                return f"No schedule found for {teacher}."
            return "Which teacher's schedule would you like to know about?"
        elif intent == "course_availability":
            course = entities.get("course") or self.context["last_course"]
            if course:
                if course in self.normalized_courses.values():
                    return f"✅ {course} is available this semester."
                return f"❌ {course} is not available this semester."
            return "Which course are you checking?"
        elif intent == "section_info":
            course = entities.get("course") or self.context["last_course"]
            if course:
                sections = [c for c in self.normalized_courses.values() if course in c]
                if sections:
                    return f"📚 Available sections: {', '.join(sections)}"
                return f"No sections found for {course}."
            return "Which course's sections would you like?"
        elif intent == "academic_calendar":
            return "📅 Academic calendar: Please check the university website for exact dates."
        elif intent == "smalltalk_weather":
            return random.choice(["Hava durumu? ☀️ Check your window or a weather app!", "Weather? I’m just predict sunny vibes! 😎"])
        elif intent == "smalltalk_name":
            return "I'm Unibot, your friendly university schedule assistant! 😊"
        return "Sorry, I didn’t understand. Could you clarify?"

def main():
    st.set_page_config(page_title="University Chatbot", layout="centered")

    st.title("🎓 University Chatbot")

    # Initialize chatbot
    if 'chatbot' not in st.session_state:
        try:
            st.session_state.chatbot = UniversityChatbot('cleaned_schedule.xlsx')
        except Exception as e:
            st.error(f"Chatbot başlatılamadı: {e}")
            return

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display conversation
    for msg in st.session_state.messages:
        with st.chat_message("user"):
            st.markdown(msg["user"])
        with st.chat_message("assistant"):
            st.markdown(msg["bot"])

    # User input
    prompt = st.chat_input("Sorunuzu yazınız...")

    if prompt:
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Process input
        try:
            entities = st.session_state.chatbot.extract_entities(prompt)
            response = st.session_state.chatbot.generate_response(prompt, entities)
        except Exception as e:
            response = f"⚠️ Hata: {e}"

        # Display bot response
        with st.chat_message("assistant"):
            st.markdown(response)

        # Save to history
        st.session_state.messages.append({
            "user": prompt,
            "bot": response
        })

if __name__ == "__main__":
    main()