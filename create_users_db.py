import os
import sqlite3

db_path = "users.db"
if os.path.exists(db_path):
    os.remove(db_path)
    print(f"Removed corrupted {db_path}")

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Create users table
cursor.execute('''
    CREATE TABLE users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        email TEXT NOT NULL UNIQUE,
        password TEXT NOT NULL
    )
''')

# Create notes table with user_note_id
cursor.execute('''
    CREATE TABLE notes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        user_note_id INTEGER NOT NULL,
        summary TEXT NOT NULL,
        keywords TEXT,
        questions TEXT,
        flashcards TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id)
    )
''')

conn.commit()
conn.close()
print(f"Created new {db_path} with 'users' and 'notes' tables")