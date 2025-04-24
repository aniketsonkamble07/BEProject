import spacy
import string
import os
from transcriber import transcribe_video_to_chunks  # Assuming you have this function in transcriber.py
from nltk.parse.stanford import StanfordParser
from nltk.tree import Tree
import pprint
import json

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

# Initialize lists that were previously undefined
sent_list = []
sent_list_detailed = []
word_list = []
word_list_detailed = []
final_words = []
final_words_detailed = []
final_output_in_sent = []
final_words_dict = {}

# Preprocess a single chunk of text
def preprocess_text(text):
    doc = nlp(text)
    cleaned_tokens = []
    for token in doc:
        # Token is not a stop word, punctuation, or space
        if token.text.lower() not in nlp.Defaults.stop_words and token.text not in string.punctuation and not token.is_space:
            cleaned_tokens.append(token.lemma_.lower())
    return cleaned_tokens

# Path to the video
video_path = r"E:\BEProject\static\video\Water.mp4"  # Use raw string literal for Windows path

# Ensure the video file exists
if not os.path.exists(video_path):
    raise FileNotFoundError(f"❌ Video not found at: {video_path}")
# Function to handle noun clauses (e.g., noun phrases or subjects)
def handle_noun_clause(i, tree_traversal_flag, modified_parse_tree, sub_tree):
    print(f"Handling noun clause: {sub_tree.text}")
    if hasattr(sub_tree.head, 'i') and sub_tree.head.i in tree_traversal_flag:
        print(f"Head token {sub_tree.head.text} is in tree_traversal_flag.")
        if tree_traversal_flag[sub_tree.i] == 0 and tree_traversal_flag[sub_tree.head.i] == 0:
            tree_traversal_flag[sub_tree.i] = 1
            modified_parse_tree.insert(i, sub_tree)
            print(f"Inserted noun clause at position {i}: {sub_tree.text}")
            i += 1
    else:
        print(f"Warning: Head token {sub_tree.head.text} is not in tree_traversal_flag.")
    return i, modified_parse_tree

# Function to handle verb or prepositional clauses (and recursively check for noun clauses)
def handle_verb_prop_clause(i, tree_traversal_flag, modified_parse_tree, sub_tree):
    print(f"Handling verb/prep clause: {sub_tree.text}")
    if hasattr(sub_tree.head, 'i') and sub_tree.head.i in tree_traversal_flag:
        print(f"Head token {sub_tree.head.text} is in tree_traversal_flag.")
        for child_sub_tree in sub_tree.subtree:
            print(f"Processing child subtree: {child_sub_tree.text}")
            if child_sub_tree.dep_ == "nsubj" or child_sub_tree.dep_ == 'prep':
                if tree_traversal_flag[child_sub_tree.i] == 0 and tree_traversal_flag[child_sub_tree.head.i] == 0:
                    tree_traversal_flag[child_sub_tree.i] = 1
                    modified_parse_tree.insert(i, child_sub_tree)
                    print(f"Inserted verb/prep clause at position {i}: {child_sub_tree.text}")
                    i += 1
    else:
        print(f"Warning: Head token {sub_tree.head.text} is not in tree_traversal_flag.")
    return i, modified_parse_tree

# Function to label and parse subtrees in the document
def label_parse_subtrees(doc):
    tree_traversal_flag = {}
    modified_parse_tree = []
    i = 0

    for token in doc:
        print(f"Token: {token.text}, Position: {token.i}, Dependency: {token.dep_}, Head: {token.head.text}")
        tree_traversal_flag[token.i] = 0

        if token.dep_ == "nsubj" or token.dep_ == "np":
            print(f"Found noun phrase (nsubj/np): {token.text}")
            i, modified_parse_tree = handle_noun_clause(i, tree_traversal_flag, modified_parse_tree, token)
        
        if token.dep_ == "VERB" or token.dep_ == "prep":
            print(f"Found verb/prep (VERB/prep): {token.text}")
            i, modified_parse_tree = handle_verb_prop_clause(i, tree_traversal_flag, modified_parse_tree, token)

    return tree_traversal_flag, modified_parse_tree

def modify_tree_structure(parent_tree):
    print("Modifying tree structure...")
    tree_traversal_flag, _ = label_parse_subtrees(parent_tree)
    modified_parse_tree = Tree('ROOT', [])
    i = 0

    for sub_tree in parent_tree.subtrees():
        print(f"Processing subtree: {sub_tree.label()}")
        if sub_tree.label() == "NP":
            i, modified_parse_tree = handle_noun_clause(i, tree_traversal_flag, modified_parse_tree, sub_tree)
        if sub_tree.label() == "VP" or sub_tree.label() == "PRP":
            i, modified_parse_tree = handle_verb_prop_clause(i, tree_traversal_flag, modified_parse_tree, sub_tree)

    for sub_tree in parent_tree.subtrees():
        for child_sub_tree in sub_tree.subtrees():
            print(f"Processing child subtree: {child_sub_tree.text}")
            if len(child_sub_tree.leaves()) == 1:
                if tree_traversal_flag[child_sub_tree.treeposition()] == 0 and tree_traversal_flag[child_sub_tree.parent().treeposition()] == 0:
                    tree_traversal_flag[child_sub_tree.treeposition()] = 1
                    modified_parse_tree.insert(i, child_sub_tree)
                    print(f"Inserted child subtree at position {i}: {child_sub_tree.text}")
                    i = i + 1

    return modified_parse_tree
from nltk.parse.stanford import StanfordParser
from nltk.tree import ParentedTree

# Assuming you have the correct StanfordParser path set in your environment
def reorder_eng_to_isl(input_string):
    count = 0
    for word in input_string:
        if len(word) == 1 and word.isalpha():
            count += 1

    if count == len(input_string):
        print("All input is single characters.")
        return input_string

    # Initialize StanfordParser (ensure your Stanford parser path is set correctly)
    parser = StanfordParser()

    # Parse the input string using the Stanford Parser
    possible_parse_tree_list = [tree for tree in parser.parse(input_string)]
    print("Testing possible parse trees:", possible_parse_tree_list)
    
    if not possible_parse_tree_list:
        print("No parse trees found.")
        return []

    parse_tree = possible_parse_tree_list[0]
    parent_tree = ParentedTree.convert(parse_tree)

    print("Original parse tree:", parse_tree)
    
    # Modify tree structure (this involves handling noun/verb clauses, etc.)
    modified_parse_tree = modify_tree_structure(parent_tree)

    # Get the leaves from the modified tree as the final parsed sentence
    parsed_sent = modified_parse_tree.leaves()
    print("Parsed sentence:", parsed_sent)
    return parsed_sent

# Function to convert the final words list into a list of individual letters if needed
def convert_to_final(final_words):
    final_output_in_sent = []
    for word in final_words:
        print(f"Processing word: {word}")
        final_output_in_sent.append(final_output(word))  # Assuming final_output is defined elsewhere
    return final_output_in_sent

# checks if sigml file exists of the word, if not, use letters for the words
def final_output(word):
    print(f"Checking word: {word}")
    # Open the file and read valid words once
    with open("words.txt", 'r') as file:
        valid_words = file.read().splitlines()  # .splitlines() handles newline characters properly

    fin_words = []
    word = word.lower()  # Convert the word to lowercase for case-insensitive comparison

    if word in valid_words:
        # If the word is valid, add it as-is
        print(f"Word '{word}' is valid.")
        fin_words.append(word)
    else:
        # If the word is not valid, break it down into individual letters
        print(f"Word '{word}' is not valid, breaking it down into letters.")
        fin_words.extend(list(word))  # Using extend to append the letters individually

    return fin_words


# Print function for displaying lists
def print_lists():
    print("--------------------Word List------------------------")
    pprint.pprint(word_list)
    print("--------------------Final Words------------------------")
    pprint.pprint(final_words)
    print("---------------Final sentence with letters--------------")
    pprint.pprint(final_output_in_sent)


# Clears all the lists after completing the work
def clear_all():
    sent_list.clear()
    sent_list_detailed.clear()
    word_list.clear()
    word_list_detailed.clear()
    final_words.clear()
    final_words_detailed.clear()
    final_output_in_sent.clear()
    final_words_dict.clear()


# Dict for sending data to front end in JSON
final_words_dict = {}


# Function to populate the dictionary with processed data
def populate_final_words_dict():
    final_words_dict['processed_words'] = final_words
    final_words_dict['final_sentence'] = final_output_in_sent


# Function to convert the dictionary into a JSON string
def get_json_data():
    return json.dumps(final_words_dict)


from flask import Flask, render_template, send_from_directory, jsonify
import os
import json

app = Flask(__name__)

# Initialize lists
sent_list = []
sent_list_detailed = []
word_list = []
word_list_detailed = []
final_words = []
final_words_detailed = []
final_output_in_sent = []
final_words_dict = {}

# Preprocess function to handle text
def preprocess_text(text):
    # Add logic for text preprocessing, e.g., lemmatization, tokenization
    return text.lower().split()

# Example video transcription function (replace with your actual implementation)
def transcribe_video_to_chunks(video_path):
    # Replace this with your actual transcription logic
    return ["This", "is", "a", "sample", "transcript"]

def generate_sigml(text, filepath):
    content = f"<sigml><hns_sign gloss='{text}'/></sigml>"
    with open(filepath, 'w') as f:
        f.write(content)

def get_word_sigml_mapping(transcript_text):
    base_path = "static/SignFiles"
    timeline = []
    words = transcript_text.lower().split()
    current_time = 0

    for word in words:
        sigml_filename = f"{word}.sigml"
        full_path = os.path.join(base_path, sigml_filename)

        if os.path.exists(full_path):
            timeline.append({
                "time": current_time,
                "file": f"/{full_path.replace(os.sep, '/')}"} )
            current_time += 3  # Increment time for each word
        else:
            print(f"⚠️ Missing file for word: {word}")

    return timeline

@app.route('/static/<path:path>')
def serve_signfiles(path):
    return send_from_directory('static', path)

@app.route('/video')
def video_page():
    return render_template('videoplayer.html')

@app.route('/sigml')
def sigml_player():
    return render_template('sigmlplayer.html')  

@app.route("/get_sigml_timeline", methods=["GET"])
def get_sigml_timeline():
    video_path = 'static/video/Water.mp4'

    if not os.path.exists(video_path):
        return jsonify({"error": "❌ Video not found"}), 404

    try:
        transcript_chunks = transcribe_video_to_chunks(video_path)
        transcript_text = " ".join(transcript_chunks)
        timeline = get_word_sigml_mapping(transcript_text)
        return jsonify(timeline)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/videoPlayer')
def video_player():
    video_path = 'static/video/sampledvdo2.mp4'

    if not os.path.exists(video_path):
        return "❌ Video not found", 404

    try:
        # Clear previous data
        clear_all()

        # Step 1: Transcribe video
        transcript_chunks = transcribe_video_to_chunks(video_path)
        transcript_text = " ".join(transcript_chunks)

        # Step 2: Process the transcript input (calls other processing functions like convert())
        take_input(transcript_text)

        # Step 3: Generate the SIGML timeline from processed final_output_in_sent
        timeline = []
        for word_group in final_output_in_sent:
            for word in word_group:
                sigml_filename = f"{word}.sigml"
                full_path = os.path.join("static/SignFiles", sigml_filename)
                if os.path.exists(full_path):
                    timeline.append({
                        "file": f"/{full_path.replace(os.sep, '/')}",
                        "time": len(timeline) * 3  # Increment time for each sign
                    })
                else:
                    print(f"⚠️ Missing file for word: {word}")

        # Step 4: Save timeline to JSON file
        with open('static/js/sigmlFiles.json', 'w') as f:
            json.dump(timeline, f)

        # Step 5: Clear all after saving
        clear_all()

        # Render the play page with the SIGML timeline
        return render_template('play.html', sigml_timeline=timeline)

    except Exception as e:
        return f"❌ Error: {e}", 500

# Clear all lists after work is done
def clear_all():
    sent_list.clear()
    sent_list_detailed.clear()
    word_list.clear()
    word_list_detailed.clear()
    final_words.clear()
    final_words_detailed.clear()
    final_output_in_sent.clear()
    final_words_dict.clear()

# Function for processing input text (your logic for converting transcript)
def take_input(transcript_text):
    # Add the text processing logic here (convert, preprocess, final output generation)
    final_words.extend(transcript_text.split())  # Example of filling the list
    final_output_in_sent.extend(final_words)

# Start Flask app
if __name__ == '__main__':
    app.run(debug=True)
