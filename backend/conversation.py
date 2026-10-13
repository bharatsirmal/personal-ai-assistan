"""Language and response policy helpers; no network or model initialization."""
import re
from datetime import datetime
from zoneinfo import ZoneInfo

LANGUAGES = {"en": "English", "hi": "Hindi", "ne": "Nepali"}


def resolve_language(question, preferred="en"):
    q = question.lower()
    # Only a request to speak/change language changes the preference. A question
    # about Hindi/Nepali as a subject should not switch an English conversation.
    command = re.search(r"speak|talk|reply|respond|answer|switch|भाषा|बोल|बात|जवाफ|जवाब|बताओ|b[oau]l|baat", q)
    if command and not re.search(r"don't|do not|नहीं.*बोल|नबोल", q):
        if re.search(r"nepali|नेपाली", q):
            return "ne"
        if re.search(r"hindi|हिन्दी|हिंदी", q):
            return "hi"
        if re.search(r"english|अंग्रेजी|अंग्रेज़ी|अङ्ग्रेजी", q):
            return "en"
    if re.search(r"[\u0900-\u097f]", q):
        if re.search(r"(?<![\u0900-\u097f])(?:छ|छन्|छौ|हुन्छ|गर्छ|गर्न|कस्तो|तपाईं|तिमी|मेरो|उसको|बताउनु)(?![\u0900-\u097f])", q):
            return "ne"
        if re.search(r"क्या|कैसे|है|हैं|बताओ|करेंगे", q):
            return "hi"
        if preferred == "ne":
            return "ne"
        return "hi"
    if re.search(r"\b(kya|kaise|batao|bataiye|hai|hain|hoga|kamzori)\b", q):
        return "hi"
    if re.search(r"\b(chha|cha|kasari|kasto|mero|tapai|garcha)\b", q):
        return "ne"
    return preferred if preferred in LANGUAGES else "en"


def needs_translation(question, language):
    return language != "en" or bool(re.search(r"[\u0900-\u097f]", question))


def is_analysis(question):
    return bool(re.search(
        r"\b(if|would|could|should|handle|weakness\w*|strength\w*|prove|proof|evidence|"
        r"confiden\w*|interview|score|rate|rating|ready|readiness|suitab\w*|scenario|"
        r"situation\w*|pressure|deadline|fail\w*|improv\w*|recommend|assess\w*)\b",
        question.lower(),
    ))


def clock_context():
    india = datetime.now(ZoneInfo("Asia/Kolkata"))
    nepal = india.astimezone(ZoneInfo("Asia/Kathmandu"))
    return (f"Current India date/time: {india:%A, %Y-%m-%d %H:%M:%S} IST (Asia/Kolkata). "
            f"Current Nepal date/time: {nepal:%A, %Y-%m-%d %H:%M:%S} NPT (Asia/Kathmandu). "
            "Use India time by default; explicitly label the timezone. Dates are Gregorian.")
