"""Checks the sentence engine on the sign sequences a patient actually makes.

Two things are checked for every case:
  1. no sign is silently dropped (the failure the old engine had);
  2. the sentence reads as expected, where we can state that exactly.

    python test_sentences.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import sentence as S

# (glosses, expected sentence). None means "no exact expectation, just check
# that nothing was dropped and it reads as a sentence".
CASES = [
    # greetings
    (["hello"], "Hello!"),
    (["hello", "how_are_you"], "Hello! How are you?"),
    (["thank_you"], "Thank you."),
    # one symptom
    (["cough"], "I have a cough."),
    (["tired"], "I feel tired."),
    (["hot"], "I have a fever."),
    (["trouble_breathing"], "I have trouble breathing."),
    # several symptoms, grouped by have/feel
    (["headache", "cough"], "I have a headache and a cough."),
    (["fever", "cough", "tired"], "I have a fever and a cough and feel tired."),
    (["nausea", "tired"], "I feel nauseous and tired."),
    # symptom + time, and tense from the time word
    (["i", "sick", "today"], "I am sick today."),
    (["sick", "yesterday"], "I was sick yesterday."),
    (["good", "tomorrow"], "I will be better tomorrow."),
    (["i", "hot", "night"], "I have a fever at night."),
    (["mother", "fever", "yesterday"], "My mother had a fever yesterday."),
    # states
    (["you", "healthy"], "You are healthy."),
    (["he", "dead"], "He is dead."),
    (["young", "strong"], "I am young and strong."),
    # places
    (["i", "hospital"], "I am going to the hospital."),
    (["i", "hospital", "yesterday"], "I was at the hospital yesterday."),
    (["i", "bed"], "I am in bed."),
    (["i", "bathroom"], "I am in the bathroom."),
    (["hospital", "tomorrow"], "I will go to the hospital tomorrow."),
    # things
    (["i", "medicine"], "I need medicine."),
    (["i", "medicine", "money"], None),          # both must appear
    # combinations the old engine dropped words from
    (["i", "family", "hospital"], None),
    (["i", "friend", "hospital", "today"], None),
    (["i", "bed", "tired"], None),
    (["medicine", "bad"], None),
    (["she", "patient", "hospital"], None),
    (["i", "cough", "night", "weak"], None),
    (["patient", "bed", "weak"], None),
    # people
    (["father", "doctor"], None),
    (["doctor", "medicine", "morning"], None),
    # verbs and question words (not taught yet, but the grammar is ready)
    (["i", "want", "water"], "I want water."),
    (["i", "need", "help"], None),
    (["i", "go", "hospital", "tomorrow"], "I will go to the hospital tomorrow."),
    (["doctor", "give", "medicine"], "The doctor gives medicine."),
    (["i", "not", "understand"], None),
    (["yes"], "Yes."),
    (["no"], "No."),
    (["please", "help"], None),
    (["where", "bathroom"], None),
    # numbers from fingerspelling
    (["i", "need", "2", "tablet"], None),
    # body parts: the symptom is worded around the part it names
    (["pain", "head"], "I have a headache."),
    (["pain", "chest"], "I have pain in my chest."),
    (["bleeding", "nose"], "I have bleeding in my nose."),
    (["my", "head", "pain", "2", "week"], "I have a headache for 2 weeks."),
    # severity reads differently on a noun and on an adjective
    (["very", "weak"], "I am very weak."),
    (["little", "pain"], "I have a little pain."),
    (["swelling", "leg", "very"], "I have a lot of swelling in my leg."),
    # how long it has been going on
    (["i", "pain", "stomach", "3", "day"], "I have a stomach ache for 3 days."),
    (["i", "vomit", "2", "day"], "I have been vomiting for 2 days."),
    # can / cannot
    (["i", "cannot", "sleep"], "I cannot sleep."),
    (["i", "can", "walk"], "I can walk."),
    (["i", "not", "can", "eat"], "I cannot eat."),
    # questions the doctor asks back
    (["where", "pain"], "Where is the pain?"),
    (["how_many", "day"], "How many days?"),
    # what the doctor does, and what the patient needs
    (["doctor", "give", "injection"], "The doctor gives an injection."),
    # needing is happening now even when the test is not, so WANT and NEED keep
    # the present tense next to a future time word
    (["i", "need", "test", "tomorrow"], "I need a test tomorrow."),
    (["i", "need", "water", "please"], "Please, I need water."),
    (["i", "better", "today"], "I am better today."),
    (["i", "worse", "night"], "I am worse at night."),
    # round 2: what the doctor says
    (["sit"], "Please sit down."),
    (["open_mouth"], "Please open your mouth."),
    (["show", "knee"], "Please show me your knee."),
    (["not", "stand"], "Please do not stand up."),
    # round 2: what the patient brings with them
    (["i", "diabetes"], "I have diabetes."),
    (["i", "diabetes", "blood_pressure"], "I have diabetes and high blood pressure."),
    (["i", "pregnant"], "I am pregnant."),
    (["i", "allergy", "medicine"], "I have an allergy to medicine."),
    (["i", "diabetes", "pain", "chest"], "I have diabetes and pain in my chest."),
    # round 2: symptoms that sit on a body part, with the right verb
    (["i", "numb", "finger"], "I feel numb in my finger."),
    (["i", "cramp", "leg"], "I have cramps in my leg."),
    (["i", "rash", "back"], "I have a rash in my back."),
    (["i", "pain", "neck", "3", "day"], "I have pain in my neck for 3 days."),
    # round 2: who the signer is, and what they need
    (["excuse_me", "i", "deaf"], "Excuse me. I am deaf."),
    (["i", "deaf", "i", "use", "sign_language"], "I am deaf and use sign language."),
    (["i", "need", "interpreter"], "I need an interpreter."),
    (["i", "want", "appointment", "tomorrow"], "I want an appointment tomorrow."),
    (["where", "pharmacy"], "Where is the pharmacy?"),
    (["age", "25"], "I am 25 years old."),
    (["i", "afraid", "operation"], "I am afraid of an operation."),
    (["ok"], "Okay."),
    (["goodbye"], "Goodbye."),
    # the INCLUDE words added in round 2, which train from the dataset
    (["doctor", "see", "i", "monday"], "The doctor sees me on Monday."),
    (["doctor", "give", "i", "medicine"], "The doctor gives me medicine."),
    (["i", "give", "you", "money"], "I give you money."),
    (["grandmother", "hospital", "sunday"], "My grandmother is going to the hospital on Sunday."),
    (["man", "sick"], "The man is sick."),
    (["woman", "pain", "chest"], "The woman has pain in her chest."),
    (["i", "wait", "10", "minute"], "I am waiting for 10 minutes."),
    (["swelling", "big_large"], "I have big swelling."),
    (["i", "swelling", "leg", "big_large"], "I have big swelling in my leg."),
    (["good_morning"], "Good morning."),
    (["alright"], "All right."),
]

# Words that may legitimately not appear literally in the sentence, because they
# are rendered as something else.
RENDERED_AS = {
    "hot": "fever", "cold": "cold", "bad": "not well", "good": ["better", "good"],
    "i": ["I", "my", "me"], "he": "He", "she": "She", "you": ["You", "your"],
    "house": "home", "how_are_you": "How are you", "thank_you": "Thank you",
    "hello": "Hello", "not": "not", "please": "Please", "stomach_ache": "stomach ache",
    "sore_throat": "sore throat", "runny_nose": "runny nose", "nausea": "nauseous",
    "trouble_breathing": "trouble breathing", "yes": "Yes", "no": "No",
    "help": "help", "where": "Where", "go": ["going", "go", "went"],
    "give": ["give", "gives", "gave"], "want": ["want", "wants"],
    "need": ["need", "needs"], "understand": "understand", "medicine": "medicine",
    "family": "family", "friend": "friend", "patient": "patient", "tablet": "tablet",
    # the new hospital vocabulary
    "pain": ["pain", "ache"], "head": ["head", "headache"], "my": ["my", "I"],
    "stomach": ["stomach", "stomach ache"], "very": ["very", "a lot of"],
    "little": ["little", "a little"], "how_many": "How many", "can": ["can", "cannot"],
    "cannot": "cannot", "walk": "walk", "sleep": "sleep", "eat": "eat",
    "water": "water", "test": "test", "injection": "injection", "vomit": "vomiting",
    "bleeding": "bleeding", "swelling": "swelling", "leg": "leg", "chest": "chest",
    "nose": "nose", "day": ["day", "days"], "week": ["week", "weeks"],
    "better": "better", "worse": "worse",
    # round 2
    "sit": "sit down", "stand": "stand up", "open_mouth": "open your mouth",
    "show": "show me", "knee": "knee", "diabetes": "diabetes",
    "blood_pressure": "blood pressure", "pregnant": "pregnant",
    "allergy": "allergy", "numb": "numb", "cramp": "cramps", "rash": "rash",
    "finger": "finger", "neck": "neck", "back": "back", "deaf": "deaf",
    "excuse_me": "Excuse me", "use": "use", "sign_language": "sign language",
    "interpreter": "interpreter", "appointment": "appointment",
    "pharmacy": "pharmacy", "age": ["years old", "I am"], "afraid": "afraid",
    "operation": "operation", "ok": "Okay", "goodbye": "Goodbye",
    # INCLUDE round 2
    "monday": "Monday", "sunday": "Sunday", "minute": "minutes",
    "grandmother": "grandmother", "man": "man", "woman": "woman",
    "big_large": "big", "good_morning": "Good morning", "alright": "All right",
    "see": ["sees", "see"], "wait": "waiting", "money": "money",
}


def covered(gloss, text):
    """Does the sentence account for this sign?"""
    lower = text.lower()
    expected = RENDERED_AS.get(gloss, gloss.replace("_", " "))
    options = expected if isinstance(expected, list) else [expected]
    return any(o.lower() in lower for o in options)


def main():
    exact_ok = exact_total = 0
    dropped = []
    for glosses, expected in CASES:
        text, _ = S.build(glosses, use_llm=False)
        missing = [g for g in glosses if not covered(g, text)]
        if missing:
            dropped.append((glosses, text, missing))
        if expected is not None:
            exact_total += 1
            if text == expected:
                exact_ok += 1
            else:
                print(f"  different: {' '.join(glosses)}")
                print(f"      wanted: {expected}")
                print(f"      got   : {text}")

    print(f"\nexact matches : {exact_ok}/{exact_total}")
    print(f"cases checked : {len(CASES)}")
    if dropped:
        print(f"\nsigns dropped in {len(dropped)} case(s):")
        for glosses, text, missing in dropped:
            print(f"  {' '.join(glosses):<34} -> {text}    missing: {missing}")
    else:
        print("no sign was dropped in any case")

    untaught_used = sorted({g for glosses, _ in CASES for g in glosses
                            if g in S.LEXICON and not S.LEXICON[g][1].get('taught')})
    print(f"\ncases include {len(untaught_used)} signs not yet taught: {untaught_used}")
    return len(dropped) == 0 and exact_ok == exact_total


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
