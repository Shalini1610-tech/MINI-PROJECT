# init_db.py
import sqlite3

conn = sqlite3.connect('contacts.db')
c = conn.cursor()

c.execute('''CREATE TABLE IF NOT EXISTS contacts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    email TEXT NOT NULL,
    message TEXT NOT NULL
)''')

conn.commit()
conn.close()
