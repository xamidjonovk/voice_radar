import sqlite3

# Connect to the database (create a new one if it doesn't exist)
conn = sqlite3.connect('database.db')

# Create a cursor object to execute SQL commands
cursor = conn.cursor()

# Create the "people" table with modified schema
cursor.execute("""CREATE TABLE IF NOT EXISTS people (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               first_name TEXT DEFAULT NULL,
               last_name TEXT DEFAULT NULL,
               folder TEXT DEFAULT NULL,
               about TEXT DEFAULT NULL,
               photo_path TEXT DEFAULT NULL
               )""")
# Add an index on the "folder" field
cursor.execute('CREATE INDEX IF NOT EXISTS idx_folder ON people (folder)')

# Commit the changes and close the connection
conn.commit()
conn.close()
