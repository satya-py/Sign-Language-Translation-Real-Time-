"""Glosses -> one English sentence.

Indian Sign Language drops the grammar words English needs: "I SICK TODAY" is a
complete ISL sentence, and the translation has to put back "am". So this module
takes the recognised signs, sorts them into slots, and renders one sentence.

Why slots and not a list of patterns: with 46 signs there are thousands of
combinations, and the previous pattern-matching version silently dropped whatever
did not fit the pattern it matched ("I FAMILY HOSPITAL" came out as "I am with my
family", losing the hospital). A slot filler uses every sign it is given.

    subject   who it is about            (default: the patient, "I")
    negation  NOT
    verb      WANT, NEED, GO, GIVE ...
    symptoms  FEVER, COUGH, TIRED ...    grouped by have/feel
    states    SICK, WEAK, GOOD ...
    places    HOSPITAL, BED ...
    things    MEDICINE, WATER ...
    people    DOCTOR, MOTHER ...         as a companion, when not the subject
    numbers   spelled digits from fingerspelling
    time      TODAY, YESTERDAY ...       also sets the tense

Vocabulary and wording live in vocab.json, so teaching the model a new sign only
needs one line there. An LLM, when a key is set, does the same job with better
phrasing; these rules are the offline fallback and the safety net.
"""

import json
import os
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).parent
VOCAB_PATH = HERE / "vocab.json"

def _load(path):
    with open(path, encoding="utf-8") as handle:
        return {k: v for k, v in json.load(handle).items() if not k.startswith("_")}


VOCAB = _load(VOCAB_PATH)
# vocab_extra.json is the rest of the hospital vocabulary, vocab_round2.json what an
# appointment needs on top of it. Any vocab_*.json found here is merged, so adding
# signs stays a matter of one file and no code.
for _path in sorted(HERE.glob("vocab_*.json")):
    for _kind, _entries in _load(_path).items():
        VOCAB.setdefault(_kind, {}).update(_entries)

# word -> (kind, entry), built once
LEXICON = {}
for _kind, _entries in VOCAB.items():
    for _word, _entry in _entries.items():
        LEXICON[_word] = (_kind, _entry)

KIND = {word: kind for word, (kind, _) in LEXICON.items()}
UNTAUGHT = sorted(w for w, (_, e) in LEXICON.items() if not e.get("taught", False))

SYSTEM = (
    "You turn Indian Sign Language glosses into ONE natural English sentence for a "
    "conversation in a hospital between a deaf patient and a doctor. "
    "Glosses are in signing order and have no grammar words; add only the missing "
    "ones (is, am, have, the, to, a, my, will). "
    "ISL has no separate sign for some clinical words, so read them in context: "
    "HOT = fever, COLD = chills, BAD = not feeling well, BATHROOM = needs the toilet. "
    "YESTERDAY means past tense, TOMORROW means future. "
    "Use every gloss you are given and invent nothing else - no symptoms, names, "
    "medicines or numbers that are not there. "
    "Some glosses come with alternatives in brackets; pick whichever makes the "
    "sentence sensible. Reply with the sentence only.\n"
    "Examples:\n"
    "I SICK TODAY -> I am sick today.\n"
    "I HOT NIGHT -> I have a fever at night.\n"
    "HEADACHE COUGH -> I have a headache and a cough.\n"
    "I FAMILY HOSPITAL -> I am at the hospital with my family.\n"
    "MOTHER HOSPITAL YESTERDAY -> My mother was at the hospital yesterday.\n"
    "GOOD TOMORROW -> I will feel better tomorrow.\n"
    "DOCTOR MEDICINE MORNING -> The doctor gives medicine in the morning.\n"
    "HELLO HOW_ARE_YOU -> Hello, how are you?\n"
    "THANK_YOU -> Thank you."
)


# --------------------------------------------------------------------------- LLM

def _post(url, payload, headers, timeout=8):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json", **headers})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _lattice_text(glosses, alternatives=None):
    """'I SICK[or WEAK] TODAY' - alternatives let the model fix a misread sign."""
    parts = []
    for i, gloss in enumerate(glosses):
        alts = (alternatives or {}).get(i) or (alternatives or {}).get(gloss) or []
        alts = [a for a in alts if a and a != gloss][:2]
        parts.append(gloss.upper() + (f"[or {' or '.join(a.upper() for a in alts)}]"
                                      if alts else ""))
    return " ".join(parts)


def llm_sentence(glosses, alternatives=None, context="", timeout=8):
    """A sentence from an LLM, or None when no key is set or the call fails."""
    text = _lattice_text(glosses, alternatives)
    if context:
        text = f"({context})\n{text}"
    groq = os.environ.get("GROQ_API_KEY")
    anthropic = os.environ.get("ANTHROPIC_API_KEY")
    try:
        if groq:
            out = _post("https://api.groq.com/openai/v1/chat/completions",
                        {"model": "openai/gpt-oss-20b", "temperature": 0.2, "max_tokens": 60,
                         "messages": [{"role": "system", "content": SYSTEM},
                                      {"role": "user", "content": text}]},
                        {"Authorization": f"Bearer {groq}"}, timeout)
            return out["choices"][0]["message"]["content"].strip()
        if anthropic:
            out = _post("https://api.anthropic.com/v1/messages",
                        {"model": "claude-haiku-4-5", "max_tokens": 60, "system": SYSTEM,
                         "messages": [{"role": "user", "content": text}]},
                        {"x-api-key": anthropic, "anthropic-version": "2023-06-01"}, timeout)
            return out["content"][0]["text"].strip()
    except (urllib.error.URLError, KeyError, ValueError, TimeoutError, OSError):
        return None
    return None


# ------------------------------------------------------------------- slot filling

def parse(glosses):
    """Sort the signs into slots. Unknown words are kept, never dropped."""
    slots = {"greetings": [], "answers": [], "question": None, "negate": False,
             "subject": None, "verbs": [], "symptoms": [], "states": [], "places": [],
             "things": [], "people": [], "numbers": [], "times": [], "unknown": [],
             "bodies": [], "severity": None, "durations": [], "modal": None,
             "commands": [], "conditions": [], "info": [], "objects": []}

    for gloss in [g.lower() for g in glosses if g and g != "idle"]:
        if gloss.isdigit():
            slots["numbers"].append(gloss)
            continue
        kind, entry = LEXICON.get(gloss, (None, None))
        if kind == "polite":
            slots["greetings"].append(entry)
        elif kind == "answer":
            slots["answers"].append(entry)
        elif kind == "question":
            slots["question"] = entry
        elif kind == "negation":
            slots["negate"] = True
        elif kind == "pronoun":
            if slots["subject"] is None:
                slots["subject"] = ("pronoun", gloss, entry)
            elif slots["subject"][1] != gloss:
                # DOCTOR SEE YOU: the second person named is who it is done to.
                # The same pronoun again (I ... I ...) is the signer keeping the
                # subject, not an object.
                slots["objects"].append(entry.get("object", entry["name"].lower()))
        elif kind == "person":
            if slots["subject"] is None:
                slots["subject"] = ("person", gloss, entry)
            else:
                slots["people"].append(entry)
        elif kind == "verb":
            slots["verbs"].append((gloss, entry))
        elif kind == "symptom":
            slots["symptoms"].append(entry)
        elif kind == "state":
            slots["states"].append((gloss, entry))
        elif kind == "place":
            slots["places"].append((gloss, entry))
        elif kind == "thing":
            slots["things"].append(entry)
        elif kind == "time":
            slots["times"].append(entry)
        elif kind == "body":
            slots["bodies"].append(entry)
        elif kind == "severity":
            slots["severity"] = entry
        elif kind == "duration":
            slots["durations"].append(entry)
        elif kind == "modal":
            slots["modal"] = entry
        elif kind == "info":
            slots["info"].append(entry)
        elif kind == "command":
            slots["commands"].append(entry)
        elif kind == "condition":
            slots["conditions"].append(entry)
        elif kind == "possessive":
            owner = entry["pronoun"]
            if slots["subject"] is None and owner in LEXICON:
                slots["subject"] = ("pronoun", owner, LEXICON[owner][1])
        else:
            slots["unknown"].append(gloss)
    return slots


def _tense(slots):
    for entry in slots["times"]:
        if entry.get("tense") in ("past", "future"):
            return entry["tense"]
    return "present"


def _conjugate(subject, tense):
    """(name, be, have, third_person) for the sentence subject."""
    if subject is None:                                   # nobody signed: the patient
        name, be, have, third = "I", "am", "have", False
    elif subject[0] == "pronoun":
        entry = subject[2]
        name, be, have = entry["name"], entry["be"], entry["have"]
        third = entry["be"] == "is"
    else:
        name = subject[2]["name"].capitalize() if subject[2]["name"].startswith("my") \
            else subject[2]["name"].capitalize()
        name = subject[2]["name"][0].upper() + subject[2]["name"][1:]
        be, have, third = "is", "has", True

    if tense == "past":
        be = "was" if be in ("am", "is") else "were"
        have = "had"
    elif tense == "future":
        be, have = "will be", "will have"
    return name, be, have, third


def _join(parts):
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    return ", ".join(parts[:-1]) + " and " + parts[-1]


def _with_severity(phrase, slots, noun=False):
    """VERY WEAK stays "very weak", but VERY SWELLING is "a lot of swelling"."""
    word = slots["severity"]["word"] if slots["severity"] else None
    if not word:
        return phrase
    bare = phrase.split(" ", 1)[1] if phrase.startswith(("a ", "an ")) else phrase
    if word == "a little":
        return f"a little {bare}"
    if word == "very" and noun:
        return f"a lot of {bare}"
    if word in ("more", "less") and noun:
        return f"{word} {bare}"
    return f"{word} {phrase}"


def _possessive(slots):
    """Whose body it is: the woman's chest is "her chest", not "my chest"."""
    subject = slots["subject"]
    if subject is None:
        return "my"
    if subject[0] == "pronoun":
        return subject[2].get("possessive", "my")
    name = subject[2].get("name", "")
    return "their" if not name.startswith(("my ", "the ")) else (
        "my" if name.startswith("my ") else _GENDER.get(subject[1], "their"))


# Which possessive reads naturally for the people the vocabulary names.
_GENDER = {"doctor": "their", "patient": "their", "nurse": "their", "man": "his",
           "woman": "her", "boy": "his", "girl": "her", "baby": "their"}


def _body_phrase(slots):
    """PAIN HEAD -> 'a headache'; BLEEDING NOSE -> 'bleeding in my nose'."""
    if not slots["bodies"]:
        return None
    body = slots["bodies"][0]
    locatable = [e for e in slots["symptoms"] if e.get("locatable")]
    if not locatable:
        return None
    symptom = locatable[0]
    if symptom["phrase"] == "pain" and body.get("ache"):
        return body["ache"]
    return f"{symptom['phrase']} in {_possessive(slots)} {body['name']}"


ALLERGIC = "allergy"


def _condition_clause(slots, have, be):
    """DIABETES, PREGNANT: what the patient lives with, not today's complaint.

    These come first in the sentence because a doctor needs them before the
    symptoms, and PREGNANT is an adjective in English while the rest are nouns.
    """
    if not slots["conditions"]:
        return []
    states = [e["state"] for e in slots["conditions"] if e.get("state")]
    nouns = [e["name"] for e in slots["conditions"] if not e.get("state")]
    out = []
    # ALLERGY MEDICINE: the thing named is what the patient reacts to, so it belongs
    # in this clause instead of becoming "and I need medicine".
    if "an allergy" in nouns and slots["things"]:
        to = _join([e["name"] for e in slots["things"]])
        nouns[nouns.index("an allergy")] = f"an allergy to {to}"
        slots["things"] = []
    if nouns:
        out.append(f"{have} {_join(nouns)}")
    if states:
        out.append(f"{be} {_join(states)}")
    return out


def _symptom_clause(slots, have, be, third):
    located = _body_phrase(slots)
    placed = slots["symptoms"][0] if located and slots["symptoms"] else None
    if placed is not None and not placed.get("locatable"):
        placed = next((e for e in slots["symptoms"] if e.get("locatable")), None)
    has_parts = [e["phrase"] for e in slots["symptoms"]
                 if e["verb"] == "have" and e is not placed]
    feel_parts = [e["phrase"] for e in slots["symptoms"]
                  if e["verb"] == "feel" and e is not placed]
    # NUMB FINGER is felt, not had: the body part follows whichever verb the
    # symptom uses, so it never comes out as "have numb in my finger".
    if located:
        if placed is not None and placed["verb"] == "feel":
            feel_parts.insert(0, located)
        else:
            has_parts.insert(0, located)
    clauses = []
    if has_parts:
        has_parts = [_with_severity(has_parts[0], slots, noun=True)] + has_parts[1:]
        clauses.append(f"{have} {_join(has_parts)}")
    if feel_parts:
        if have.startswith("will"):
            feel = "will feel"
        elif have == "had":
            feel = "felt"
        else:
            feel = "feels" if third else "feel"
        clauses.append(f"{feel} {_join(feel_parts)}")
    return clauses


ABOUT = {"afraid": "of", "worried": "about", "angry": "about"}


def _state_clause(slots, be):
    # BIG, SMALL, SLOW, FAST describe the symptom next to them - SWELLING BIG is
    # "big swelling", not "I have swelling and I am big".
    if slots["symptoms"]:
        sizes = [e["phrase"] for _g, e in slots["states"] if e.get("describes_symptom")]
        if sizes:
            slots["states"] = [(g, e) for g, e in slots["states"]
                               if not e.get("describes_symptom")]
            first = slots["symptoms"][0]
            phrase = first["phrase"]
            bare = phrase.split(" ", 1)[1] if phrase.startswith(("a ", "an ")) else phrase
            article = "a " if phrase.startswith(("a ", "an ")) else ""
            slots["symptoms"][0] = {**first, "phrase": f"{article}{' '.join(sizes)} {bare}"}

    phrases = [entry["phrase"] for _gloss, entry in slots["states"]]
    if not phrases:
        return []
    # AFRAID OPERATION is fear of the operation, not two separate facts.
    linked = next((g for g, _ in slots["states"] if g in ABOUT), None)
    if linked and slots["things"]:
        about = _join([f"{e.get('article', 'the ')}{e['name']}" for e in slots["things"]])
        phrases = [f"{p} {ABOUT[linked]} {about}" if p == dict(
            (g, e["phrase"]) for g, e in slots["states"])[linked] else p for p in phrases]
        slots["things"] = []
    phrases = [_with_severity(phrases[0], slots)] + phrases[1:]
    return [f"{be} {_join(phrases)}"]


def _place_clause(slots, be, tense, moving):
    """In bed, or going to the hospital: BED is where you are, HOSPITAL is where you go."""
    clauses = []
    for gloss, entry in slots["places"]:
        static = gloss in ("bed", "bathroom") and not moving
        if static:
            clauses.append(f"{be} {entry['at']}")
        elif tense == "past":
            clauses.append(f"went {entry['to']}" if moving else f"{be} {entry['at']}")
        elif tense == "future":
            clauses.append(f"will go {entry['to']}")
        else:
            going = "is going" if be == "is" else ("are going" if be == "are" else "am going")
            clauses.append(f"{going} {entry['to']}")
    return clauses


def _thing_clause(slots, tense, third):
    """A thing with no verb is something the person needs."""
    if not slots["things"]:
        return []
    names = []
    for entry in slots["things"]:
        count = slots["numbers"][0] if slots["numbers"] else None
        names.append(f"{count} {entry['name']}" if count else
                     f"{entry.get('article', '')}{entry['name']}")
    # A patient with a thing needs it; a doctor or nurse with a thing gives it.
    giver = slots["subject"] and slots["subject"][1] in ("doctor", "nurse")
    if giver:
        verb = {"past": "gave", "future": "will give"}.get(tense, "gives")
    else:
        verb = {"past": "needed", "future": "will need"}.get(tense, "needs" if third else "need")
    return [f"{verb} {_join(names)}"]


def _verb_clause(slots, tense, third):
    clauses = []
    # "The doctor gives ME medicine": whoever it is done to comes first, and it is
    # not joined to the thing with "and" - that would read "gives me and medicine".
    recipients = list(slots["objects"])
    objects = []
    counts = list(slots["numbers"])
    for entry in slots["things"]:
        count = counts.pop(0) if counts else None      # "2 tablet" -> "2 tablets"
        if count:
            plural = entry["name"] if entry["name"].endswith("s") else entry["name"] + "s"
            objects.append(f"{count} {plural if count not in ('1', '0') else entry['name']}")
        else:
            objects.append(f"{entry.get('article', '')}{entry['name']}")
    for _gloss, entry in slots["places"]:
        objects.append(entry["to"])
    for entry in slots["people"]:
        objects.append(entry["name"])

    for gloss, verb in slots["verbs"]:
        form = verb.get({"past": "past", "future": "future"}.get(tense, "present"))
        # "I want an appointment tomorrow" - wanting is happening now even though
        # the appointment is not, so WANT and NEED keep the present with a future time.
        if tense == "future" and gloss in ("want", "need"):
            form = verb.get("present", form)
        if tense == "present" and third:
            form = verb.get("third", form)
        clause = form
        if recipients:
            clause += " " + _join(recipients)
            recipients = []
        if objects:
            clause += " " + _join(objects)
            objects = []                       # objects belong to the first verb
        clauses.append(clause)
    return clauses


def rule_sentence(glosses):
    """Build the sentence from slots. Every gloss given is used somewhere."""
    slots = parse(glosses)
    lead = [e["fixed"] for e in slots["greetings"] if e.get("fixed") and not e.get("inline")]
    lead += [e["fixed"] for e in slots["answers"]]

    body_signs = (slots["subject"] or slots["verbs"] or slots["symptoms"] or slots["states"]
                  or slots["places"] or slots["things"] or slots["people"]
                  or slots["unknown"] or slots["times"] or slots["bodies"]
                  or slots["durations"] or slots["modal"] or slots["severity"]
                  or slots["commands"] or slots["conditions"] or slots["info"]
                  or slots["objects"])
    if not body_signs:
        question = slots["question"]
        if question and question.get("fixed"):
            lead.append(question["fixed"])
        return " ".join(lead).strip()

    if slots["info"] and not slots["question"]:
        entry = slots["info"][0]
        if entry.get("unit"):                      # AGE 25 -> "I am 25 years old."
            count = slots["numbers"][0] if slots["numbers"] else None
            if count:
                return " ".join(lead + [f"{entry['phrase']} {count} {entry['unit']}."]).strip()
            return " ".join(lead + [f"{entry['phrase']} ... {entry['unit']}."]).strip()
        spelled = _join([w.replace("_", " ") for w in slots["unknown"]])
        return " ".join(lead + [f"{entry['phrase']} {spelled or '...'}".strip() + "."]).strip()

    if slots["commands"] and slots["subject"] is None and not slots["question"]:
        asked = _join([e["verb"] for e in slots["commands"]])
        if slots["bodies"]:
            # SHOW HAND: the body part is what to show, not the seat of a symptom
            asked += " your " + _join([e["name"] for e in slots["bodies"]])
        if slots["negate"]:
            return " ".join(lead + [f"Please do not {asked}."]).strip()
        return " ".join(lead + [f"Please {asked}."]).strip()

    tense = _tense(slots)
    name, be, have, third = _conjugate(slots["subject"], tense)
    moving = any(v[1].get("motion") for v in slots["verbs"]) or bool(slots["verbs"])

    clauses = []
    if slots["verbs"]:
        clauses += _state_clause(slots, be)
        clauses += _verb_clause(slots, tense, third)
    else:
        clauses += _condition_clause(slots, have, be)
        state_clauses = _state_clause(slots, be)      # may reword a symptom first
        clauses += _symptom_clause(slots, have, be, third)
        clauses += state_clauses
        clauses += _place_clause(slots, be, tense, moving=False)
        clauses += _thing_clause(slots, tense, third)

    # people who are not the subject travel with the sentence
    if slots["people"] and not slots["verbs"]:
        companions = _join([e["name"] for e in slots["people"]])
        if clauses:
            clauses[-1] += f" with {companions}"
        else:
            clauses.append(f"{be} with {companions}")

    if len(clauses) > 1:
        # Two clauses that both open with the same verb read as one list.
        merged = [clauses[0]]
        for clause in clauses[1:]:
            first = clause.split(" ", 1)
            previous = merged[-1].split(" ", 1)[0]
            if len(first) == 2 and first[0] == previous:
                merged[-1] += f" and {first[1]}"
            else:
                merged.append(clause)
        clauses = merged

    if slots["objects"] and not slots["verbs"]:
        clauses.append(f"with {_join(slots['objects'])}")

    if slots["unknown"]:
        clauses.append(_join([w.replace('_', ' ') for w in slots["unknown"]]))

    if not clauses:
        clauses = [f"{be} here"]

    if slots["modal"]:
        clauses = [_modalise(c, slots["modal"], slots["negate"]) for c in clauses]
    elif slots["negate"]:
        clauses = [_negate(c, be, have, third, tense) for c in clauses]

    tail = ""
    if slots["durations"]:                      # "for 3 days"
        count = slots["numbers"][0] if slots["numbers"] else None
        unit = slots["durations"][0]
        word = unit["many"] if count and count != "1" else unit["one"]
        tail += f" for {count + ' ' if count else 'a '}{word}"
    if slots["times"]:
        tail += " " + _join([e["phrase"] for e in slots["times"]])
    please = any(e.get("inline") for e in slots["greetings"])
    body = f"{name} {' and '.join(clauses)}{tail}".strip()
    if please:
        head = body.split(" ", 1)[0]
        rest = body[len(head):]
        body = "Please, " + (head if head == "I" else head.lower()) + rest

    question = slots["question"]
    if question and question.get("fixed"):
        return " ".join(lead + [question["fixed"]]).strip()
    if question:
        return " ".join(lead + [_question(question, slots, name, be)]).strip()

    return " ".join(lead + [body + "."]).strip()


BASE_FORM = {"feels": "feel", "felt": "feel", "needs": "need", "needed": "need",
             "wants": "want", "wanted": "want", "gives": "give", "gave": "give",
             "sees": "see", "saw": "see", "takes": "take", "took": "take",
             "understands": "understand", "understood": "understand",
             "went": "go", "came": "come", "ate": "eat", "drank": "drink",
             "slept": "sleep", "called": "call"}


def _question(question, slots, name, be):
    """WHERE BATHROOM is "Where is the bathroom?", not "Where I am in the bathroom?"."""
    word = question["word"]
    if slots["durations"] and not slots["symptoms"]:
        unit = slots["durations"][0]
        return f"{word} {unit['many'] if word.endswith('many') else unit['one']}?"
    if slots["symptoms"] and slots["subject"] is None:
        phrase = slots["symptoms"][0]["phrase"]
        bare = phrase.split(" ", 1)[1] if phrase.startswith(("a ", "an ")) else phrase
        return f"{word} is the {bare}?"
    if slots["bodies"] and slots["subject"] is None:
        return f"{word} is the {slots['bodies'][0]['name']}?"
    if slots["places"] and slots["subject"] is None:
        noun = slots["places"][0][1].get("noun", slots["places"][0][0])
        return f"{word} is {noun}?"
    if slots["things"] and slots["subject"] is None:
        thing = slots["things"][0]
        if thing["name"] == "time":            # "WHAT TIME" is asking the clock
            return f"{word} time?"
        if word == "What":                     # "WHAT MEDICINE" asks which one
            return f"Which {thing['name']}?"
        return f"{word} is {thing.get('article', 'the ')}{thing['name']}?"
    if slots["subject"] is not None and not slots["symptoms"] and not slots["states"]:
        return f"{word} {be} {name[0].lower() + name[1:] if name != 'I' else name}?"
    body = f"{name} {be}"
    return f"{word} {body}?"


CONJUGATED = {"am going": "go", "is going": "go", "are going": "go",
              "am coming": "come", "is coming": "come", "am waiting": "wait",
              "have finished": "finish", "has finished": "finish"}


def _modalise(clause, modal, negate):
    """CAN SLEEP / CANNOT SLEEP: the modal replaces the conjugated verb."""
    word = "cannot" if (negate or modal.get("negated")) else modal["positive"]
    for conjugated, plain in CONJUGATED.items():
        if clause.startswith(conjugated):
            return f"{word} {plain}{clause[len(conjugated):]}"
    first = clause.split(" ", 1)[0]
    rest = clause[len(first):]
    if first in ("am", "is", "are", "was", "were", "have", "has", "had"):
        return f"{word} be{rest}"
    return f"{word} {BASE_FORM.get(first, first)}{rest}"


def _negate(clause, be, have, third, tense):
    """NOT applies to the verb: 'am sick' -> 'am not sick', 'understand' -> 'do not understand'."""
    first = clause.split(" ", 1)[0]
    rest = clause[len(first):]
    if first in (be, have) or first.startswith("will") or first in ("am", "is", "are",
                                                                    "was", "were", "has",
                                                                    "have", "had"):
        return f"{first} not{rest}"
    helper = {"past": "did not", "future": "will not"}.get(
        tense, "does not" if third else "do not")
    return f"{helper} {BASE_FORM.get(first, first)}{rest}"


def build(glosses, use_llm=True, alternatives=None, context=""):
    """Returns (sentence, source) where source is 'llm' or 'rules'."""
    glosses = [g for g in glosses if g and g != "idle"]
    if not glosses:
        return "", "rules"
    if use_llm:
        out = llm_sentence(glosses, alternatives=alternatives, context=context)
        if out:
            return out, "llm"
    return rule_sentence(glosses), "rules"


def spelled_sentence(word, kind="name"):
    """A fingerspelled word turned into a sentence."""
    pretty = word[:1].upper() + word[1:].lower()
    return {"name": f"My name is {pretty}.",
            "place": f"I live in {pretty}.",
            "medicine": f"My medicine is {pretty}."}.get(kind, pretty)


if __name__ == "__main__":
    import test_sentences
    test_sentences.main()
