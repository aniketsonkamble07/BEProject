import json
import logging
import os
import re
import string
from venv import logger
import stanza
from nltk.parse.stanford import StanfordParser
from nltk.tree import *
import zipfile
import sys
import time
import ssl
from flask import Flask, request, render_template, send_from_directory, jsonify
import urllib
from transcriber import transcribe_video_to_chunks
import pprint
import string
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from nltk.corpus import wordnet as wn
import numpy as np

# Initialize Flask app
app = Flask(__name__, static_folder='static', static_url_path='')

# SSL configuration
ssl._create_default_https_context = ssl._create_unverified_context

# Base directory setup
BASE_DIR = os.path.dirname(os.path.realpath(__file__))
os.environ['CLASSPATH'] = os.path.join(BASE_DIR, 'stanford-parser-full-2018-10-17')
os.environ['STANFORD_MODELS'] = os.path.join(BASE_DIR, 'stanford-parser-full-2018-10-17/edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz')
os.environ['NLTK_DATA'] = '/usr/local/share/nltk_data/'

from nltk.corpus import stopwords
# Initialize NLP pipeline
stanza.download('en', model_dir='stanza_resources')
en_nlp = stanza.Pipeline('en', processors={'tokenize': 'spacy'})
nltk_stopwords = set(stopwords.words('english'))


sent_list = []
sent_list_detailed = []
word_list = []
word_list_detailed = []
final_words = []
final_words_detailed = []
final_output_in_sent = []
final_words_dict = {}

# Helper functions
def is_parser_jar_file_present():
    stanford_parser_zip_file_path = os.environ.get('CLASSPATH') + ".jar"
    return os.path.exists(stanford_parser_zip_file_path)

def reporthook(count, block_size, total_size):
    global start_time
    if count == 0:
        start_time = time.perf_counter()
        return
    duration = time.perf_counter() - start_time
    progress_size = int(count * block_size)
    speed = int(progress_size / (1024 * duration))
    percent = min(int(count*block_size*100/total_size),100)
    sys.stdout.write("\r...%d%%, %d MB, %d KB/s, %d seconds passed" %
                    (percent, progress_size / (1024 * 1024), speed, duration))
    sys.stdout.flush()

def download_parser_jar_file():
    stanford_parser_zip_file_path = os.environ.get('CLASSPATH') + ".jar"
    url = "https://nlp.stanford.edu/software/stanford-parser-full-2018-10-17.zip"
    urllib.request.urlretrieve(url, stanford_parser_zip_file_path, reporthook)

def extract_parser_jar_file():
    stanford_parser_zip_file_path = os.environ.get('CLASSPATH') + ".jar"
    try:
        with zipfile.ZipFile(stanford_parser_zip_file_path) as z:
            z.extractall(path=BASE_DIR)
    except Exception:
        os.remove(stanford_parser_zip_file_path)
        download_parser_jar_file()
        extract_parser_jar_file()

def extract_models_jar_file():
    stanford_models_zip_file_path = os.path.join(os.environ.get('CLASSPATH'), 'stanford-parser-3.9.2-models.jar')
    stanford_models_dir = os.environ.get('CLASSPATH')
    with zipfile.ZipFile(stanford_models_zip_file_path) as z:
        z.extractall(path=stanford_models_dir)

def download_required_packages():
    if not os.path.exists(os.environ.get('CLASSPATH')):
        if not is_parser_jar_file_present():
            download_parser_jar_file()
        extract_parser_jar_file()

    if not os.path.exists(os.environ.get('STANFORD_MODELS')):
        extract_models_jar_file()

# Core processing functions
def convert_to_sentence_list(text):
    print("\n=== convert_to_sentence_list() ===")
    print(f"INPUT text type: {type(text)}")
    print(f"Sample text: {text.text[:100]}..." if hasattr(text, 'text') else str(text)[:100])
    
    sent_list.clear()
    sent_list_detailed.clear()
    for i, sentence in enumerate(text.sentences, 1):
        sent_list.append(sentence.text)
        sent_list_detailed.append(sentence)
        if i <= 3:  # Print first 3 sentences as samples
            print(f"Sentence {i}: {sentence.text[:50]}...")
    
    print(f"\nOUTPUT: {len(sent_list)} sentences processed")
    print(f"Sample output sentence: {sent_list[0][:50]}..." if sent_list else "No sentences processed")
    print("=== End convert_to_sentence_list() ===\n")
def convert_to_word_list(sentences):
    word_list.clear()
    word_list_detailed.clear()
    temp_list = []
    temp_list_detailed = []
    
    for sentence in sentences:
        for word in sentence.words:
            temp_list.append(word.text)
            temp_list_detailed.append(word)
        
        word_list.append(temp_list.copy())
        word_list_detailed.append(temp_list_detailed.copy())
        temp_list.clear()
        temp_list_detailed.clear()


def get_stop_words():
    """Return a comprehensive set of English stop words"""
    base_stopwords = set(stopwords.words('english'))
    
    extended_stopwords = {
        # Verbs
        'is', 'are', 'was', 'were', 'be', 'being', 'been', 'have', 'has', 'had',
        'do', 'does', 'did', 'can', 'could', 'shall', 'should', 'will', 'would',
        'may', 'might', 'must', 'ought',
        
        # Articles/determiners
        'the', 'a', 'an', 'this', 'that', 'these', 'those',
        
        # Conjunctions
        'and', 'or', 'but', 'nor', 'yet', 'so', 'for', 'as',
        
        # Prepositions
        'at', 'by', 'for', 'in', 'of', 'on', 'to', 'with', 'from', 'into', 'about',
        'above', 'after', 'against', 'along', 'among', 'around', 'before', 'behind',
        'below', 'beneath', 'beside', 'between', 'beyond', 'during', 'except',
        'inside', 'near', 'off', 'over', 'past', 'since', 'through', 'toward',
        'under', 'until', 'upon', 'within', 'without',
        
        # Pronouns
        'i', 'me', 'my', 'myself', 'we', 'us', 'our', 'ours', 'ourselves',
        'you', 'your', 'yours', 'yourself', 'yourselves',
        'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself',
        'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
        
        # Others
        'there', 'here', 'when', 'where', 'why', 'how', 'what', 'which', 'who', 'whom',
        'whose', 'not', 'no', 'nor', 'only', 'own', 'same', 'so', 'than', 'too',
        'very', 'some', 'such', 'other', 'all', 'any', 'both', 'each', 'few', 'more',
        'most', 'other', 'some', 'such', 'none',
        
        # Contractions
        "'s", "'re", "'ve", "'d", "'ll", "'m", "n't", "'t"
    }
    
    return base_stopwords.union(extended_stopwords)
def filter_words(word_list):
    """
    Filter out stop words from a list of word groups.
    
    Args:
        word_list: List of lists containing words to be filtered
        
    Returns:
        List of lists with stop words removed
    """
    stop_words = get_stop_words()
    
    if not word_list:
        return []
    
    # Enhanced stop words list
    extended_stopwords = nltk_stopwords.union({
        "'s", "'re", "'ve", "'d", "'ll", "'m",  # contractions
        "n't", "'t",                             # negations
        "one", "ones", "something", "anything"    # pronouns
    })
    
    return [
        [word for word in group if word.lower() not in extended_stopwords]
        for group in word_list
    ]


def remove_punct(word_list):
    """
    Remove punctuation marks from a list of word groups.
    
    Args:
        word_list: List of lists containing words to be processed
        
    Returns:
        None (modifies the input list in-place)
    """
    if not word_list:
        return
    
    # Comprehensive punctuation set including Unicode punctuation
    punct_set = set(string.punctuation).union({
        '“', '”', '‘', '’', '—', '–', '…', '«', '»'
    })
    
    for i, group in enumerate(word_list):
        word_list[i] = [word for word in group if word not in punct_set]


def lemmatize(final_word_list):
    for words, final in zip(word_list_detailed, final_word_list):
        for i, (word, fin) in enumerate(zip(words, final)):
            word_text = str(word.text) if word.text else ""
            fin_text = str(fin) if fin else ""
            
            if fin_text.lower() == word_text.lower() and len(fin_text) > 1:
                final[i] = word.lemma


def label_parse_subtrees(parent_tree):
    return {sub_tree.treeposition(): 0 for sub_tree in parent_tree.subtrees()}

def handle_noun_clause(i, tree_traversal_flag, modified_parse_tree, sub_tree):
    print(f"\nProcessing NP at position {sub_tree.treeposition()}")
    
    if tree_traversal_flag[sub_tree.treeposition()] == 0 and tree_traversal_flag[sub_tree.parent().treeposition()] == 0:
        tree_traversal_flag[sub_tree.treeposition()] = 1
        modified_parse_tree.insert(i, sub_tree)
        i += 1
        print(f"Added NP subtree. New index: {i}")
    else:
        print("Skipping NP subtree (already processed or parent processed)")
    
    return i, modified_parse_tree

def handle_verb_prop_clause(i, tree_traversal_flag, modified_parse_tree, sub_tree):
    print(f"\nProcessing {'VP' if sub_tree.label() == 'VP' else 'PRP'} at position {sub_tree.treeposition()}")
    
    added_count = 0
    for child_sub_tree in sub_tree.subtrees():
        if child_sub_tree.label() in ["NP", "PRP"]:
            if tree_traversal_flag[child_sub_tree.treeposition()] == 0 and tree_traversal_flag[child_sub_tree.parent().treeposition()] == 0:
                tree_traversal_flag[child_sub_tree.treeposition()] = 1
                modified_parse_tree.insert(i, child_sub_tree)
                i += 1
                added_count += 1
    
    print(f"Added {added_count} children from this {'VP' if sub_tree.label() == 'VP' else 'PRP'}")
    return i, modified_parse_tree

def modify_tree_structure(parent_tree):
    print("\n=== Starting tree modification ===")
    print(f"Original tree structure:\n{parent_tree.pformat()}")
    
    tree_traversal_flag = label_parse_subtrees(parent_tree)
    modified_parse_tree = Tree('ROOT', [])
    i = 0
    
    for sub_tree in parent_tree.subtrees():
        if sub_tree.label() == "NP":
            i, modified_parse_tree = handle_noun_clause(i, tree_traversal_flag, modified_parse_tree, sub_tree)
        if sub_tree.label() in ["VP", "PRP"]:
            i, modified_parse_tree = handle_verb_prop_clause(i, tree_traversal_flag, modified_parse_tree, sub_tree)
    
    single_leaf_added = 0
    for sub_tree in parent_tree.subtrees():
        for child_sub_tree in sub_tree.subtrees():
            if len(child_sub_tree.leaves()) == 1:
                if tree_traversal_flag[child_sub_tree.treeposition()] == 0 and tree_traversal_flag[child_sub_tree.parent().treeposition()] == 0:
                    tree_traversal_flag[child_sub_tree.treeposition()] = 1
                    modified_parse_tree.insert(i, child_sub_tree)
                    i += 1
                    single_leaf_added += 1
    
    print(f"\nAdded {single_leaf_added} single-leaf subtrees")
    print(f"\nFinal modified tree structure (size: {len(modified_parse_tree)}):")
    print(modified_parse_tree.pformat())
    print("=== Tree modification complete ===")
    
    return modified_parse_tree

def reorder_eng_to_isl(input_string):
    print("\n=== reorder_eng_to_isl() ===")
    print(f"Input string: {input_string}")
    
    # Check if all words are single characters (likely already processed)
    if all(len(word) == 1 for word in input_string):
        print("All words are single characters - returning input unchanged")
        return input_string
    
    # Download required NLP packages
    download_required_packages()
    
    # Parse the input string
    parser = StanfordParser()
    possible_parse_tree_list = [tree for tree in parser.parse(input_string)]
    
    if not possible_parse_tree_list:
        print("Warning: No parse trees generated - returning original input")
        return input_string
    
    parse_tree = possible_parse_tree_list[0]
    print("\nSelected parse tree:")
    print(parse_tree.pformat())
    
    # Convert to parented tree for easier navigation
    parent_tree = ParentedTree.convert(parse_tree)
    
    # Modify tree structure for ISL
    modified_parse_tree = modify_tree_structure(parent_tree)
    print("\nModified parse tree:")
    print(modified_parse_tree.pformat())
    
    # Extract leaves (words) in new order
    output_words = modified_parse_tree.leaves()
    print("\nFinal reordered words:")
    print(output_words)
    print("=== End reorder_eng_to_isl() ===")
    
    return output_words

def pre_process(text):
    print("\n=== Pre-processing text ===")
    
    # Remove punctuation
    original_counts = [len(group) for group in word_list]
    remove_punct(word_list)
    new_counts = [len(group) for group in word_list]
    print(f"Punctuation removed (counts: {original_counts} -> {new_counts})")

    # Filter stopwords
    filtered_words = filter_words(word_list)
    removed = sum(len(group) for group in word_list) - sum(len(group) for group in filtered_words)
    print(f"Removed {removed} stopwords")
    
    final_words.extend(filtered_words)
    
    # Lemmatize words
    lemmatize(final_words)
    print(f"Sample lemmatized output: {final_words[0][:5] if final_words else 'Empty'}")
# Initialize word embedding model at the module level (top of your script)
from sentence_transformers import SentenceTransformer
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def find_related_words(target_word, vocabulary, top_n=3):
    """Find related words using WordNet and semantic similarity"""
    print(f"\nFinding related words for '{target_word}'")
    
    # Get WordNet synonyms
    synonyms = set()
    for syn in wn.synsets(target_word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name().replace('_', ' '))
    
    valid_synonyms = [word for word in synonyms if word in vocabulary]
    if valid_synonyms:
        return [(word, 1.0) for word in valid_synonyms[:top_n]]
    
    # Use semantic similarity if no direct synonyms
    vocab_embeddings = embedding_model.encode(list(vocabulary))
    target_embedding = embedding_model.encode([target_word])
    similarities = cosine_similarity(target_embedding, vocab_embeddings)[0]
    top_indices = np.argsort(similarities)[-top_n:][::-1]
    return [(vocabulary[i], similarities[i]) for i in top_indices if similarities[i] > 0.5]

def final_output(input_words):
    """Convert words to ISL format with similar-word fallback"""
    print("\n=== Generating final output with similar-word fallback ===")
    
    # Load vocabulary and available signs
    try:
        with open("words.txt", 'r', encoding='utf-8') as f:
            valid_words = {line.strip().lower() for line in f if line.strip()}
        print(f"Loaded dictionary with {len(valid_words)} words")
    except FileNotFoundError:
        print("⚠️ Dictionary file not found")
        valid_words = set()

    # Get available signs
    sign_dir = "static/SignFiles"
    available_signs = {f.split('.')[0].lower() for f in os.listdir(sign_dir) 
                     if f.endswith('.sigml')} if os.path.exists(sign_dir) else set()

    # Initialize semantic model (only if needed)
    model = None
    
    fin_words = []
    stats = {
        'exact_match': 0,
        'plural_form': 0,
        'similar_meaning': 0,
        'kept_with_sign': 0,
        'kept_without_sign': 0,
        'punctuation_skipped': 0,
        'stopwords_removed': 0
    }

    for word in input_words:
        if not isinstance(word, str) or not word.strip():
            continue
            
        word_lower = word.lower().strip()
        
        # Skip punctuation
        if word in string.punctuation:
            stats['punctuation_skipped'] += 1
            continue
            
        # Skip stop words
        if word_lower in get_stop_words():
            stats['stopwords_removed'] += 1
            continue
            
        # Check if whole-word sign exists
        sigml_path = f"static/SignFiles/{word_lower}.sigml"
        has_sign = os.path.exists(sigml_path)
        
        # Strategy 1: Exact dictionary match with sign
        if word_lower in valid_words and has_sign:
            fin_words.append(word_lower)
            stats['exact_match'] += 1
            print(f"[Exact] {word_lower}")
            continue
            
        # Strategy 2: Plural form with sign
        singular = word_lower.rstrip('s')
        singular_sigml = f"static/SignFiles/{singular}.sigml"
        if (word_lower.endswith('s') and 
            singular in valid_words and 
            os.path.exists(singular_sigml)):
            fin_words.append(singular)
            stats['plural_form'] += 1
            print(f"[Plural] {word_lower} → {singular}")
            continue
            
        # Strategy 3: Any word with existing sign
        if has_sign:
            fin_words.append(word_lower)
            stats['kept_with_sign'] += 1
            print(f"[Existing sign] {word_lower}")
            continue
            
        # Strategy 4: Find similar-meaning word with sign
        similar_word = find_similar_with_sign(word_lower, available_signs)
        if similar_word:
            fin_words.append(similar_word)
            stats['similar_meaning'] += 1
            print(f"[Similar] {word_lower} → {similar_word}")
            continue
            
        # Final fallback: Keep whole word without splitting
        fin_words.append(word_lower)
        stats['kept_without_sign'] += 1
        print(f"[No sign] Keeping whole: {word_lower}")

    print("\nConversion statistics:")
    for stat, count in stats.items():
        print(f"  {stat.replace('_', ' ').title():<20}: {count}")
    
    return fin_words

def find_similar_with_sign(word, available_signs):
    """Find similar word that has a sign available"""
    # 1. Check WordNet synonyms
    for syn in wn.synsets(word):
        for lemma in syn.lemmas():
            lemma_word = lemma.name().replace('_', ' ').lower()
            if lemma_word in available_signs:
                return lemma_word
    
    # 2. Check spelling variations
    variations = [
        word + 's',  # plural
        word[:-1] if word.endswith('s') else None,  # singular
        word + 'ing',
        word + 'ed',
        word + 'er'
    ]
    for var in variations:
        if var and var in available_signs:
            return var
    
    # 3. Semantic similarity (lazy-load model)
    if not available_signs:
        return None
        
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    
    model = SentenceTransformer('all-MiniLM-L6-v2')
    target_embedding = model.encode([word])
    sign_words = list(available_signs)
    sign_embeddings = model.encode(sign_words)
    
    similarities = cosine_similarity(target_embedding, sign_embeddings)[0]
    best_idx = np.argmax(similarities)
    
    if similarities[best_idx] > 0.6:  # Similarity threshold
        return sign_words[best_idx]
    
    return None

def convert_to_final():
    print("\n=== Converting to final ISL format ===")
    final_output_in_sent.clear()
    
    for words in final_words:
        processed_words = final_output(words)
        final_output_in_sent.append(processed_words)
    
    print(f"Processed {len(final_output_in_sent)} sentences")

def take_input(text):
    print("\n=== Processing input text ===")
    print(f"Original input: {text[:50]}...")
    
    # Clean and prepare text
    test_input = text.strip().replace("\n", "").replace("\t", "")
    if len(test_input) > 1:
        test_input2 = " .".join(word.capitalize() for word in test_input.split(".")) + " ."
    
    # NLP processing
    some_text = en_nlp(test_input2)
    convert(some_text)

def convert(some_text):
    print("\n=== Converting text to ISL ===")
    
    # Process through pipeline
    convert_to_sentence_list(some_text)
    convert_to_word_list(sent_list_detailed)
    
    # Reorder words
    for i, words in enumerate(word_list):
        word_list[i] = reorder_eng_to_isl(words)
    
    # Final processing
    pre_process(some_text)
    convert_to_final()
    remove_punct(final_output_in_sent)
    
    print("\nFinal output:")
    print_lists()

def print_lists():
    """Print all processing stages with clear formatting"""
    print("\n=== Processing Results Summary ===\n")
    
    # Word List section
    print(f"1. Initial Word List ({len(word_list)} sentences):")
    for i, sent in enumerate(word_list, 1):
        print(f"   {i}. {' '.join(sent)}")
    
    # Final Words section
    print(f"\n2. Processed Words ({len(final_words)} sentences):")
    for i, sent in enumerate(final_words, 1):
        print(f"   {i}. {' '.join(sent)}")
    
    # Final Output section
    print(f"\n3. ISL Output ({len(final_output_in_sent)} sentences):")
    for i, sent in enumerate(final_output_in_sent, 1):
        print(f"   {i}. {' '.join(sent)}")
    
    print("\n=== End of Results ===")

def clear_all():
    print("\n=== clear_all() ===")
    print("Clearing all processing lists...")
    
    lists_to_clear = {
        'sent_list': sent_list,
        'sent_list_detailed': sent_list_detailed,
        'word_list': word_list,
        'word_list_detailed': word_list_detailed,
        'final_words': final_words,
        'final_words_detailed': final_words_detailed,
        'final_output_in_sent': final_output_in_sent,
        'final_words_dict': final_words_dict
    }
    
    for name, lst in lists_to_clear.items():
        before = len(lst)
        lst.clear()
        print(f"Cleared {name} (had {before} items)")
    
    print("All lists cleared")
    print("=== End clear_all() ===")

# Web routes
@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('index.html')

@app.route('/static/<path:path>')
def serve_signfiles(path):
    return send_from_directory('static', path)



@app.route('/sigml')
def sigml_player():
    return render_template('sigmlplayer.html')

def generate_sigml(text, filepath):
    with open(filepath, 'w') as f:
        f.write(f"<sigml><hns_sign gloss='{text}'/></sigml>")

def get_word_sigml_mapping(transcript_text):
    timeline = []
    current_time = 0
    
    for word in transcript_text.lower().split():
        sigml_path = os.path.join("static/SignFiles", f"{word}.sigml")
        if os.path.exists(sigml_path):
            timeline.append({
                "time": current_time,
                "file": f"/{sigml_path.replace(os.sep, '/')}"
            })
            current_time += 3
    return timeline
@app.route("/get_sigml_timeline", methods=["GET"])
def get_sigml_timeline():
    """API endpoint to get SIGML timeline - only uses whole-word signs"""
    video_path = 'static/video/Water.mp4'
    if not os.path.exists(video_path):
        logger.error(f"Video not found at {video_path}")
        return jsonify({"error": "Video not found"}), 404
    
    try:
        # 1. Transcribe video
        logger.info("Starting video transcription...")
        transcript_chunks = transcribe_video_to_chunks(video_path)
        full_text = " ".join(transcript_chunks)
        logger.info(f"Original transcript: {full_text[:100]}...")
        
        # 2. Process through NLP pipeline
        take_input(full_text)
        
        # 3. Generate timeline - WHOLE WORDS ONLY
        timeline = []
        current_time = 0
        used_words = set()  # Track used words to avoid duplicates
        
        for word_group in final_output_in_sent:
            for word in word_group:
                word_lower = word.lower().strip()
                
                # Skip conditions
                if (len(word_lower) < 2 or  # Minimum word length
                    word_lower in stopwords.words('english') or
                    word in string.punctuation or
                    word_lower in used_words):
                    continue
                
                # Only process if whole-word SIGML exists
                sigml_path = os.path.join("static/SignFiles", f"{word_lower}.sigml")
                if os.path.exists(sigml_path):
                    timeline.append({
                        "word": word_lower,
                        "file": f"/{sigml_path.replace(os.sep, '/')}",
                        "time": current_time
                    })
                    current_time += 3  # 3 seconds per sign
                    used_words.add(word_lower)
                    logger.debug(f"Added sign for: {word_lower}")
                else:
                    logger.info(f"No sign available for: {word_lower} (keeping whole word)")
        
        logger.info(f"Final timeline with {len(timeline)} signs")
        return jsonify(timeline)
        
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500
@app.route('/videoPlayer')
def video_player():
    """Render video player with whole-word ISL signs only"""
    video_path = 'static/video/sampledvdo2.mp4'
    if not os.path.exists(video_path):
        logger.error(f"Video not found at {video_path}")
        return "Video not found", 404
    
    try:
        clear_all()
        logger.info("Starting strict video processing...")
        
        # 1. Transcribe video
        transcript_chunks = transcribe_video_to_chunks(video_path)
        full_text = " ".join(transcript_chunks)
        logger.info(f"Original transcript: {full_text[:100]}...")
        
        # 2. Process through NLP pipeline
        take_input(full_text)
        
        # 3. Generate timeline - WHOLE WORDS ONLY
        timeline = []
        current_time = 0
        used_words = set()
        
        for word_group in final_output_in_sent:
            for word in word_group:
                word_lower = word.lower().strip()
                
                # Skip conditions
                if (len(word_lower) < 2 or  # Minimum word length
                    word_lower in stopwords.words('english') or
                    word in string.punctuation or
                    word_lower in used_words):
                    continue
                
                # Only process if whole-word SIGML exists
                sigml_path = os.path.join("static/SignFiles", f"{word_lower}.sigml")
                if os.path.exists(sigml_path):
                    timeline.append({
                        "word": word_lower,
                        "file": f"/{sigml_path.replace(os.sep, '/')}",
                        "time": current_time
                    })
                    current_time += 3
                    used_words.add(word_lower)
                    logger.debug(f"Added sign for: {word_lower}")
                else:
                    logger.info(f"No sign available for: {word_lower}")
        
        logger.info(f"Final timeline with {len(timeline)} signs")
        
        # Save timeline
        with open('static/js/sigmlFiles.json', 'w') as f:
            json.dump(timeline, f, indent=2)
        
        return render_template('play.html',
                            sigml_timeline=timeline,
                            original_text=full_text,
                            processed_words=list(used_words))
        
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        clear_all()
        return f"Error: {str(e)}", 500
    
if __name__ == '__main__':
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    app.run(debug=True)