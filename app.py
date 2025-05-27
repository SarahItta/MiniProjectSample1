from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session, make_response
import sqlite3
import os
from transformers import T5Tokenizer, T5ForConditionalGeneration
from sentence_transformers import SentenceTransformer, util
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import re
import PyPDF2
import json
import torch
import pandas as pd

# Download NLTK data once at startup
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

app = Flask(__name__)
app.secret_key = 'mytestkey123'

app.template_folder = 'templates'
app.static_folder = 'static'

# Load fine-tuned models
tokenizer = T5Tokenizer.from_pretrained("./t5_finetuned_new")
model = T5ForConditionalGeneration.from_pretrained("./t5_finetuned_new")
sentence_model = SentenceTransformer("./minilm_finetuned_new")

def preprocess_text(text):
    text = re.sub(r'\s+', ' ', text.strip())
    text = re.sub(r'[^\w\s.,!?]', '', text)
    return text

def summarize_text(text, max_words):
    text = preprocess_text(text)
    inputs = tokenizer("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
    max_length = int(max_words * 1.3)
    min_length = max(20, int(max_words * 0.6))
    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=max_length,
        min_length=min_length,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def extract_keywords(text, num_keywords=5):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text.lower())
    words = [word for word in words if word.isalnum() and word not in stop_words]
    word_freq = nltk.FreqDist(words)
    return [word for word, _ in word_freq.most_common(num_keywords)]

def generate_questions(summary, num_questions=3):
    sentences = sent_tokenize(summary)
    questions = []
    
    for sentence in sentences[:num_questions]:
        if len(sentence.split()) > 5:
            # Use T5 to generate a question
            input_text = f"Generate a question for: {sentence}"
            inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
            question_ids = model.generate(
                inputs["input_ids"],
                max_length=50,
                num_beams=4,
                early_stopping=True
            )
            question = tokenizer.decode(question_ids[0], skip_special_tokens=True)
            
            # Use all-MiniLM-L6-v2 to validate semantic similarity
            sentence_embedding = sentence_model.encode(sentence, convert_to_tensor=True)
            question_embedding = sentence_model.encode(question, convert_to_tensor=True)
            similarity_score = util.cos_sim(sentence_embedding, question_embedding).item()
            
            if similarity_score > 0.5:
                questions.append({
                    "question": question,
                    "answer": sentence.strip().replace(".", "")
                })
    
    # Fallback: If fewer than num_questions are generated, use a simpler method
    if len(questions) < num_questions:
        for sentence in sentences[len(questions):num_questions]:
            if len(sentence.split()) > 5:
                question = sentence.strip().replace(".", "?")
                questions.append({
                    "question": question,
                    "answer": sentence.strip().replace(".", "")
                })
    
    return questions[:num_questions]

def generate_flashcards(keywords, summary, input_text, num_flashcards=3):
    flashcards = []
    input_sentences = sent_tokenize(input_text)
    
    # Refined reference questions
    reference_questions = [
        "What is photosynthesis?",
        "What is the role of chlorophyll?",
        "What are the stages of photosynthesis?"
    ]
    
    # Encode sentences and reference questions using the fine-tuned SentenceTransformer
    input_embeddings = sentence_model.encode(input_sentences, convert_to_tensor=True)
    ref_embeddings = sentence_model.encode(reference_questions, convert_to_tensor=True)
    
    used_sentences = set()
    # Step 1: Match reference questions to sentences
    for i in range(min(num_flashcards, len(reference_questions))):
        question = reference_questions[i]
        best_answer = "No relevant information found."
        best_score = -1
        question_embedding = ref_embeddings[i]
        
        for idx, sentence in enumerate(input_sentences):
            if sentence not in used_sentences:
                sentence_embedding = input_embeddings[idx]
                score = util.cos_sim(question_embedding, sentence_embedding).item()
                if score > best_score and score > 0.5:
                    best_answer = sentence.strip()
                    best_score = score
        
        if best_score > 0.5:
            # Clean up the answer formatting
            if "stages" in question.lower() and "stages" in best_answer.lower():
                best_answer = best_answer.replace("stages ", "stages: ").replace("lightdependent", "light-dependent")
            used_sentences.add(best_answer)
            flashcards.append({"front": question, "back": best_answer})
    
    # Step 2: Fallback to remaining reference questions
    if len(flashcards) < num_flashcards:
        remaining_ref_questions = reference_questions[len(flashcards):num_flashcards]
        for i, question in enumerate(remaining_ref_questions):
            best_answer = "No relevant information found."
            best_score = -1
            question_embedding = sentence_model.encode(question, convert_to_tensor=True)
            
            for idx, sentence in enumerate(input_sentences):
                if sentence not in used_sentences:
                    sentence_embedding = input_embeddings[idx]
                    score = util.cos_sim(question_embedding, sentence_embedding).item()
                    if score > best_score and score > 0.4:
                        best_answer = sentence.strip()
                        best_score = score
            
            if best_score > 0.4:
                # Clean up the answer formatting
                if "stages" in question.lower() and "stages" in best_answer.lower():
                    best_answer = best_answer.replace("stages ", "stages: ").replace("lightdependent", "light-dependent")
                used_sentences.add(best_answer)
                flashcards.append({"front": question, "back": best_answer})
    
    # Step 3: Fallback to keyword-based questions
    if len(flashcards) < num_flashcards:
        for keyword in keywords[:num_flashcards - len(flashcards)]:
            question = f"What is {keyword}?"
            best_answer = f"No definition found for {keyword}."
            best_score = -1
            keyword_embedding = sentence_model.encode(keyword, convert_to_tensor=True)
            
            for idx, sentence in enumerate(input_sentences):
                if sentence not in used_sentences:
                    sentence_embedding = input_embeddings[idx]
                    score = util.cos_sim(keyword_embedding, sentence_embedding).item()
                    if score > best_score and score > 0.4:
                        best_answer = sentence.strip()
                        best_score = score
            
            if best_score > 0.4:
                used_sentences.add(best_answer)
                flashcards.append({"front": question, "back": best_answer})
    
    # Step 4: Default fallback
    if len(flashcards) < num_flashcards:
        remaining = num_flashcards - len(flashcards)
        for i in range(len(input_sentences)):
            if len(flashcards) >= num_flashcards:
                break
            if input_sentences[i] not in used_sentences:
                question = f"Detail {len(flashcards) + 1}: What is mentioned?"
                best_answer = input_sentences[i].strip()
                # Clean up the answer formatting
                if "stages" in best_answer.lower():
                    best_answer = best_answer.replace("stages ", "stages: ").replace("lightdependent", "light-dependent")
                flashcards.append({"front": question, "back": best_answer})
                used_sentences.add(input_sentences[i])
    
    return flashcards

@app.route('/')
def home():
    return render_template('MainPage.html')

@app.route('/main')
def mainpage():
    if 'user_id' in session:
        username = session.get('username', '')
        return render_template('MainPage.html', username=username)
    return redirect('/')

@app.route('/api/signup', methods=['POST'])
def api_signup():
    data = request.get_json()
    name = data.get('name')
    email = data.get('email')
    password = data.get('password')
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE email = ?", (email,))
    if cursor.fetchone():
        conn.close()
        return jsonify({'success': False, 'message': 'Email already exists! Please login.'})
    cursor.execute("INSERT INTO users (name, email, password) VALUES (?, ?, ?)", (name, email, password))
    conn.commit()
    conn.close()
    return jsonify({'success': True, 'message': 'Signup successful! Please login.'})

@app.route('/api/login', methods=['POST'])
def api_login():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE email = ?", (email,))
    user = cursor.fetchone()
    conn.close()
    if user and user[3] == password:
        session['user_id'] = user[0]
        session['username'] = user[1]
        return jsonify({'success': True, 'message': 'Login successful!'})
    return jsonify({'success': False, 'message': 'Invalid email or password'})

@app.route('/logout')
def logout():
    session.clear()
    return redirect('/')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/generate', methods=['GET', 'POST'])
def generate_notes():
    notes_data = {'summary': '', 'key_points': [], 'keywords': [], 'questions': [], 'flashcards': []}
    error = None
    
    if 'user_id' not in session:
        print("No user_id in session, redirecting to home")
        return redirect(url_for('home'))
    
    if request.method == 'POST':
        print("POST request received")
        input_option = request.form.get('input_option')
        user_input = ""
        max_words = int(request.form.get('length', 150))
        include_questions = request.form.get('include_questions') == 'on'
        
        print(f"Input: {input_option}, Words: {max_words}, Questions: {include_questions}")
        
        if input_option == 'text':
            user_input = request.form.get('input_text', '')
        elif input_option == 'file' and 'file' in request.files:
            file = request.files['file']
            print(f"File: {file.filename if file else 'None'}")
            if file.filename.endswith('.txt'):
                user_input = file.read().decode('utf-8')
            elif file.filename.endswith('.pdf'):
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    user_input += page.extract_text() or ""
        
        print(f"Input length: {len(user_input)} chars")
        
        if not user_input.strip():
            error = "Please provide input text or upload a file."
        else:
            cleaned_input = preprocess_text(user_input)
            input_words = len(cleaned_input.split())
            print(f"Cleaned words: {input_words}")
            if input_words < 20:
                error = "Input too short. Provide at least 20 words."
            else:
                try:
                    summary = summarize_text(cleaned_input, max_words)
                    notes_data['summary'] = summary
                    print("Summary done")
                    
                    notes_data['key_points'] = sent_tokenize(summary)
                    notes_data['keywords'] = extract_keywords(cleaned_input)
                    print(f"Keywords: {notes_data['keywords']}")
                    
                    if include_questions:
                        notes_data['questions'] = generate_questions(summary)
                        notes_data['flashcards'] = generate_flashcards(notes_data['keywords'], summary, cleaned_input)
                        print(f"Questions: {len(notes_data['questions'])}, Flashcards: {len(notes_data['flashcards'])}")
                    
                    # Store in session as fallback
                    session['temp_notes'] = notes_data
                    
                    # Save to database
                    conn = sqlite3.connect('users.db')
                    cursor = conn.cursor()
                    
                    # Check if user_note_id column exists
                    cursor.execute("PRAGMA table_info(notes)")
                    columns = [info[1] for info in cursor.fetchall()]
                    use_user_note_id = 'user_note_id' in columns
                    
                    if use_user_note_id:
                        # Get the next user_note_id
                        cursor.execute("SELECT MAX(user_note_id) FROM notes WHERE user_id = ?", (session['user_id'],))
                        max_note_id = cursor.fetchone()[0]
                        next_note_id = (max_note_id or 0) + 1
                        
                        # Insert with user_note_id
                        cursor.execute('''
                            INSERT INTO notes (user_id, user_note_id, summary, keywords, questions, flashcards)
                            VALUES (?, ?, ?, ?, ?, ?)
                        ''', (
                            session['user_id'],
                            next_note_id,
                            summary,
                            ','.join(notes_data['keywords']),
                            json.dumps(notes_data['questions']),
                            json.dumps(notes_data['flashcards'])
                        ))
                    else:
                        # Insert without user_note_id (fallback for old schema)
                        cursor.execute('''
                            INSERT INTO notes (user_id, summary, keywords, questions, flashcards)
                            VALUES (?, ?, ?, ?, ?)
                        ''', (
                            session['user_id'],
                            summary,
                            ','.join(notes_data['keywords']),
                            json.dumps(notes_data['questions']),
                            json.dumps(notes_data['flashcards'])
                        ))
                    
                    conn.commit()
                    print("Data saved to database")
                    
                except sqlite3.DatabaseError as db_err:
                    error = "Database error: Unable to save notes. Please run the migration script or recreate the database."
                    print(f"Database error: {str(db_err)}")
                except Exception as e:
                    error = f"Error processing notes: {str(e)}"
                    print(f"Processing error: {str(e)}")
                finally:
                    conn.close()
    
    print("Rendering generate.html")
    response = make_response(render_template('generate.html', notes_data=notes_data, username=session.get('username', ''), error=error))
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

@app.route('/clear_notes', methods=['POST'])
def clear_notes():
    if 'user_id' not in session:
        return redirect(url_for('home'))
    session.pop('temp_notes', None)
    return redirect(url_for('generate_notes'))

@app.route('/download_notes/<int:user_note_id>')
@app.route('/download_notes')
def download_notes(user_note_id=None):
    if 'user_id' not in session:
        return redirect(url_for('home'))
    
    try:
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        
        # Check if user_note_id column exists
        cursor.execute("PRAGMA table_info(notes)")
        columns = [info[1] for info in cursor.fetchall()]
        use_user_note_id = 'user_note_id' in columns
        
        if user_note_id and use_user_note_id:
            cursor.execute("SELECT summary, keywords, questions FROM notes WHERE user_id = ? AND user_note_id = ?", (session['user_id'], user_note_id))
        elif user_note_id:
            cursor.execute("SELECT summary, keywords, questions FROM notes WHERE user_id = ? AND id = ?", (session['user_id'], user_note_id))
        else:
            cursor.execute("SELECT summary, keywords, questions FROM notes WHERE user_id = ? ORDER BY created_at DESC LIMIT 1", (session['user_id'],))
        
        note = cursor.fetchone()
        conn.close()
        
        if note:
            summary, keywords, questions = note
            questions = json.loads(questions) if questions else []
        else:
            notes_data = session.get('temp_notes', {})
            if not notes_data.get('summary'):
                return "No notes found.", 404
            summary = notes_data['summary']
            keywords = ','.join(notes_data['keywords'])
            questions = notes_data['questions']
    
        notes_content = f"# Study Notes\n\n## Summary\n{summary}\n\n## Key Points\n"
        for sentence in sent_tokenize(summary):
            notes_content += f"- {sentence}\n"
        notes_content += "\n## Keywords\n"
        for keyword in keywords.split(','):
            notes_content += f"- {keyword}\n"
        if questions:
            notes_content += "\n## Review Questions\n"
            for i, q in enumerate(questions, 1):
                notes_content += f"{i}. {q['question']}\n"
        
        response = make_response(notes_content)
        response.headers["Content-Disposition"] = f"attachment; filename=study_notes_{user_note_id or 'latest'}.md"
        response.headers["Content-Type"] = "text/markdown"
        return response
    except sqlite3.DatabaseError:
        notes_data = session.get('temp_notes', {})
        if not notes_data.get('summary'):
            return "No notes found due to database error.", 404
        summary = notes_data['summary']
        keywords = ','.join(notes_data['keywords'])
        questions = notes_data['questions']
        
        notes_content = f"# Study Notes\n\n## Summary\n{summary}\n\n## Key Points\n"
        for sentence in sent_tokenize(summary):
            notes_content += f"- {sentence}\n"
        notes_content += "\n## Keywords\n"
        for keyword in keywords.split(','):
            notes_content += f"- {keyword}\n"
        if questions:
            notes_content += "\n## Review Questions\n"
            for i, q in enumerate(questions, 1):
                notes_content += f"{i}. {q['question']}\n"
        
        response = make_response(notes_content)
        response.headers["Content-Disposition"] = "attachment; filename=study_notes.md"
        response.headers["Content-Type"] = "text/markdown"
        return response

@app.route('/download_flashcards/<int:user_note_id>')
@app.route('/download_flashcards')
def download_flashcards(user_note_id=None):
    if 'user_id' not in session:
        return redirect(url_for('home'))
    
    try:
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        
        # Check if user_note_id column exists
        cursor.execute("PRAGMA table_info(notes)")
        columns = [info[1] for info in cursor.fetchall()]
        use_user_note_id = 'user_note_id' in columns
        
        if user_note_id and use_user_note_id:
            cursor.execute("SELECT flashcards FROM notes WHERE user_id = ? AND user_note_id = ?", (session['user_id'], user_note_id))
        elif user_note_id:
            cursor.execute("SELECT flashcards FROM notes WHERE user_id = ? AND id = ?", (session['user_id'], user_note_id))
        else:
            cursor.execute("SELECT flashcards FROM notes WHERE user_id = ? ORDER BY created_at DESC LIMIT 1", (session['user_id'],))
        
        flashcards = cursor.fetchone()
        conn.close()
        
        if flashcards and flashcards[0]:
            flashcards = json.loads(flashcards[0])
        else:
            notes_data = session.get('temp_notes', {})
            flashcards = notes_data.get('flashcards', [])
            if not flashcards:
                return "No flashcards found.", 404
        
        flashcard_csv = pd.DataFrame(flashcards).to_csv(index=False)
        
        response = make_response(flashcard_csv)
        response.headers["Content-Disposition"] = f"attachment; filename=flashcards_{user_note_id or 'latest'}.csv"
        response.headers["Content-Type"] = "text/csv"
        return response
    except sqlite3.DatabaseError:
        notes_data = session.get('temp_notes', {})
        flashcards = notes_data.get('flashcards', [])
        if not flashcards:
            return "No flashcards found due to database error.", 404
        
        flashcard_csv = pd.DataFrame(flashcards).to_csv(index=False)
        
        response = make_response(flashcard_csv)
        response.headers["Content-Disposition"] = "attachment; filename=flashcards.csv"
        response.headers["Content-Type"] = "text/csv"
        return response

@app.route('/notes')
def view_notes():
    if 'user_id' not in session:
        return redirect(url_for('home'))
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    
    # Check if user_note_id column exists
    cursor.execute("PRAGMA table_info(notes)")
    columns = [info[1] for info in cursor.fetchall()]
    use_user_note_id = 'user_note_id' in columns
    
    if use_user_note_id:
        cursor.execute("SELECT user_note_id, summary, keywords, questions, flashcards, created_at FROM notes WHERE user_id = ? ORDER BY created_at DESC", (session['user_id'],))
    else:
        cursor.execute("SELECT id, summary, keywords, questions, flashcards, created_at FROM notes WHERE user_id = ? ORDER BY created_at DESC", (session['user_id'],))
    
    saved_notes = cursor.fetchall()
    conn.close()
    notes_data = []
    for note in saved_notes:
        notes_data.append({
            'user_note_id': note[0],
            'summary': note[1],
            'keywords': note[2].split(',') if note[2] else [],
            'questions': json.loads(note[3]) if note[3] else [],
            'flashcards': json.loads(note[4]) if note[4] else [],
            'created_at': note[5]
        })
    return render_template('notes.html', notes_data=notes_data, username=session.get('username', ''))

@app.route('/flashcards')
def view_flashcards():
    if 'user_id' not in session:
        return redirect(url_for('home'))
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    
    # Check if user_note_id column exists
    cursor.execute("PRAGMA table_info(notes)")
    columns = [info[1] for info in cursor.fetchall()]
    use_user_note_id = 'user_note_id' in columns
    
    if use_user_note_id:
        cursor.execute("SELECT user_note_id, flashcards, created_at FROM notes WHERE user_id = ? AND flashcards IS NOT NULL ORDER BY created_at DESC", (session['user_id'],))
    else:
        cursor.execute("SELECT id, flashcards, created_at FROM notes WHERE user_id = ? AND flashcards IS NOT NULL ORDER BY created_at DESC", (session['user_id'],))
    
    saved_flashcards = cursor.fetchall()
    conn.close()
    flashcard_sets = []
    for note in saved_flashcards:
        flashcard_sets.append({
            'user_note_id': note[0],
            'flashcards': json.loads(note[1]) if note[1] else [],
            'created_at': note[2]
        })
    return render_template('flashcards.html', flashcard_sets=flashcard_sets, username=session.get('username', ''))

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('home'))
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute("SELECT name, email FROM users WHERE id = ?", (session['user_id'],))
    user = cursor.fetchone()
    
    # Check if user_note_id column exists
    cursor.execute("PRAGMA table_info(notes)")
    columns = [info[1] for info in cursor.fetchall()]
    use_user_note_id = 'user_note_id' in columns
    
    if use_user_note_id:
        cursor.execute("SELECT user_note_id, summary, created_at FROM notes WHERE user_id = ? ORDER BY created_at DESC", (session['user_id'],))
    else:
        cursor.execute("SELECT id, summary, created_at FROM notes WHERE user_id = ? ORDER BY created_at DESC", (session['user_id'],))
    
    saved_notes = cursor.fetchall()
    conn.close()
    notes_data = [{'user_note_id': note[0], 'summary': note[1], 'created_at': note[2]} for note in saved_notes]
    return render_template('dashboard.html', username=user[0], email=user[1], saved_notes=notes_data)

@app.route('/quiz')
def quiz():
    if 'user_id' not in session:
        return redirect(url_for('home'))
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    
    # Check if user_note_id column exists
    cursor.execute("PRAGMA table_info(notes)")
    columns = [info[1] for info in cursor.fetchall()]
    use_user_note_id = 'user_note_id' in columns
    
    if use_user_note_id:
        cursor.execute("SELECT user_note_id, questions FROM notes WHERE user_id = ? AND questions IS NOT NULL ORDER BY created_at DESC", (session['user_id'],))
    else:
        cursor.execute("SELECT id, questions FROM notes WHERE user_id = ? AND questions IS NOT NULL ORDER BY created_at DESC", (session['user_id'],))
    
    notes = cursor.fetchall()
    conn.close()
    questions = []
    for note in notes:
        note_questions = json.loads(note[1]) if note[1] else []
        for q in note_questions:
            q['user_note_id'] = note[0]
            questions.append(q)
    return render_template('quiz.html', questions=questions, username=session.get('username', ''))

@app.route('/submit_quiz', methods=['POST'])
def submit_quiz():
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': 'Please log in to submit quiz.'})
    answers = request.form.to_dict()
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    
    cursor.execute("PRAGMA table_info(notes)")
    columns = [info[1] for info in cursor.fetchall()]
    use_user_note_id = 'user_note_id' in columns
    
    results = []
    for q_key, user_answer in answers.items():
        if q_key.startswith('answer_'):
            parts = q_key.replace('answer_', '').split('_')
            if len(parts) == 2:
                note_id, q_index = map(int, parts)
                if use_user_note_id:
                    cursor.execute("SELECT questions FROM notes WHERE user_id = ? AND user_note_id = ?", (session['user_id'], note_id))
                else:
                    cursor.execute("SELECT questions FROM notes WHERE user_id = ? AND id = ?", (session['user_id'], note_id))
                note = cursor.fetchone()
                if note:
                    note_questions = json.loads(note[0]) if note[0] else []
                    if q_index < len(note_questions):
                        correct_answer = note_questions[q_index]['answer'].strip()
                        # Use all-MiniLM-L6-v2 for semantic similarity
                        user_embedding = sentence_model.encode(user_answer, convert_to_tensor=True)
                        correct_embedding = sentence_model.encode(correct_answer, convert_to_tensor=True)
                        similarity_score = util.cos_sim(user_embedding, correct_embedding).item()
                        is_correct = similarity_score > 0.7
                        results.append({
                            'question': note_questions[q_index]['question'],
                            'user_answer': user_answer,
                            'correct_answer': correct_answer,
                            'is_correct': is_correct,
                            'similarity_score': similarity_score
                        })
    conn.close()
    return jsonify({'success': True, 'results': results})

if __name__ == '__main__':
    app.run(debug=True)