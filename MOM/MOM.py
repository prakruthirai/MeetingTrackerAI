import spacy
from rake_nltk import Rake
import re

# Initialize spaCy for NER and RAKE for keyword extraction
nlp = spacy.load("en_core_web_sm")
rake_nltk_var = Rake()

def read_transcript(file_path):
    """Reads and returns the content of a transcript file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()
      
def extract_speakers_directly(transcript):
    """Extract speaker names directly from the transcript."""
    pattern = r'^\w[\w\s]*:'  # Matches names followed by colons
    speakers = set(re.findall(pattern, transcript, flags=re.MULTILINE))
    return list(speaker.strip(':') for speaker in speakers)


def extract_unique_keywords(text, top_n='N'):
    """Extract top N unique keywords from the text."""
    rake_nltk_var.extract_keywords_from_text(text)
    return list(set(rake_nltk_var.get_ranked_phrases()[:top_n]))
  
def extract_action_items(doc):
    """Extract action items from the document."""
    action_items = []
    action_keywords = ['action item', 'action point', 'to do', 'task']

    for sent in doc.sents:
        sentence_text = sent.text.strip()

        # Check if the sentence contains any action keywords
        if any(keyword.lower() in sentence_text.lower() for keyword in action_keywords):
            action_items.append(sentence_text)

    return action_items

def identify_follow_up_tasks(doc):
    """Identify follow-up tasks from the document."""
    follow_up_tasks = []
    task_indicators = ['follow-up', 'complete', 'prepare', 'review', 'submit', 'send', 'action required', 'to do']

    for sent in doc.sents:
        if any(task_indicator in sent.text.lower() for task_indicator in task_indicators):
            follow_up_tasks.append(sent.text)

    return follow_up_tasks
def process_transcript_for_complete_analysis(file_path):
    raw_transcript = read_transcript(file_path)
    doc = nlp(raw_transcript)

    if ':' in raw_transcript:
        participants = extract_speakers_directly(raw_transcript)
    else:
        participants = extract_speakers_from_paragraphs(raw_transcript)

    keywords = extract_unique_keywords(raw_transcript, 10)

    key_summaries, decisions_made, agenda_items = [], [], []
    decision_keywords = ['decide', 'decision', 'agreed', 'agree', 'will', 'shall', 'resolved']
    agenda_keywords = ['agenda', 'discuss', 'plan', 'review', 'talk about']

    for sent in doc.sents:
        sentence_text = sent.text.strip()

        if any(keyword.lower() in sentence_text.lower() for keyword in keywords):
            key_summaries.append(sentence_text)

        if any(kw in sentence_text.lower() for kw in decision_keywords):
            decisions_made.append(sentence_text)

        if any(kw in sentence_text.lower() for kw in agenda_keywords):
            agenda_items.append(sentence_text)

    action_items = extract_action_items(doc)
    follow_up_tasks = identify_follow_up_tasks(doc)

    return {
        "Participants": participants,
        "Keywords": keywords,
        "KeySummaries": key_summaries,  # Include all matching sentences
        "DecisionsMade": decisions_made,
        "AgendaItems": agenda_items,
        "ActionItems": action_items,
        "FollowUpTasks": follow_up_tasks
    }



if __name__ == "__main__": 
    file_path = "transcript.txt" # Make sure this path is correct
    mom_details = process_transcript_for_complete_analysis(file_path)

    print("\nKeywords:", ", ".join(mom_details["Keywords"]))
    print("\nMinutes of the Meeting (MoM)")
    print("\nParticipants:", ", ".join(mom_details["Participants"]))
    print("\nKey Summaries:")
    for summary in mom_details["KeySummaries"]:
        print("-", summary)
    print("\nDecisions Made:")
    for decision in mom_details["DecisionsMade"]:
        print("-", decision)
    print("\nAgenda Items:")
    for item in mom_details["AgendaItems"]:
        print("-", item)
    print("\nAction Items:")
    for action_item in mom_details["ActionItems"]:
        print("-", action_item)
    print("\nFollow-Up Tasks:")
    for task in mom_details["FollowUpTasks"]:
        print("-", task)

