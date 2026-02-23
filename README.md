# Database Project — Local Recruiting Database (CSV → SQLite)

This project transforms LinkedIn / LinkedHelper CSV exports into a local SQLite database.

It is designed to:
- Keep all personal data strictly local
- Structure messy CSV files into clean tables
- Prepare the system for later AI matching (vector search + LLM)

No cloud database is used.

---

# 🚀 Quick Start (Step-by-Step)

These instructions assume:
- You have Python installed
- You have Git installed
- You cloned this repository from GitHub

---

# 1️⃣ Open the Project Folder

Open PowerShell (Windows) and navigate to the project folder:

```powershell
cd "PATH\TO\database_project"

Example:

cd "C:\Users\YourName\Desktop\database_project"
2️⃣ Create a Virtual Environment (IMPORTANT)

This isolates project dependencies.

Run:

python -m venv .venv

This creates a folder called .venv.

3️⃣ Activate the Virtual Environment

On Windows (PowerShell):

.\.venv\Scripts\Activate.ps1

If you get a security error, run once:

Set-ExecutionPolicy -Scope CurrentUser RemoteSigned

Then activate again.

After activation, your terminal should show:

(.venv) PS C:\...
4️⃣ Install Required Packages

Run:

pip install -r requirements.txt

This installs:

pandas

PyYAML

and required dependencies

🗄️ Database Setup
5️⃣ Initialize the Database (Run Once)

This creates the local database file candidates.db.

python init_db.py

You should see:

✅ DB initialized

If this step is skipped, the import will fail.

📥 Import a CSV File
6️⃣ Place Your CSV Somewhere Locally

⚠️ Do NOT put CSV files into this repository folder.
They are ignored by Git for data protection.

Just keep them somewhere on your computer.

7️⃣ Edit the CSV Path

Open import_csv.py.

At the bottom you will see:

csv_file_path = r"Database_extraction\DevOneIdent_170.csv"

Replace this path with the location of your CSV file.

Example:

csv_file_path = r"C:\Users\YourName\Downloads\linkedin_export.csv"

Save the file.

8️⃣ Run the Import
python import_csv.py

This will:

Detect the CSV delimiter automatically

Skip malformed rows

Insert new candidates

Update existing candidates (based on profile_url)

Store:

PII locally

Matching text separately

Messaging history separately

You should see something like:

✅ Import done. Inserted: 170, Updated: 0
🔍 Verify Everything Worked
9️⃣ Check the Database

Run:

python check_db.py

This prints:

All tables

All column names

Indexes

Row counts

Sample rows

Expected output example:

Tables found: candidate, candidate_profile_text, candidate_messages

candidate: 170
candidate_profile_text: 170
candidate_messages: 91

If counts look correct → import succeeded.

🔁 Importing More CSV Files Later

If you receive a new CSV export:

Replace the path in import_csv.py

Run:

python import_csv.py

The script will:

Insert new candidates

Update existing ones

Avoid duplicates (based on LinkedIn profile_url)

No need to re-run init_db.py.

📁 What Files Do What?
File	Purpose
schema.sql	Defines database structure
init_db.py	Creates database and tables
import_csv.py	Imports CSV into database
check_db.py	Verifies database content
extraction.py	Explore CSV structure
config.yml	Mapping configuration
requirements.txt	Python dependencies
🔒 Data Protection

This repository intentionally ignores:

*.csv

*.db

virtual environments

secret files

The actual candidate data stays local and is never pushed to GitHub.

🧠 What Comes Next (Future Steps)

After CSV import is stable:

Build embedding documents from candidate_profile_text

Add vector database (ChromaDB or Qdrant)

Add Streamlit dashboard

Implement 2-stage matching (semantic filter + LLM ranking)

❓ Troubleshooting

If git is not recognized:
→ Reinstall Git and select "Add to PATH"

If PowerShell blocks activation:
→ Run Set-ExecutionPolicy -Scope CurrentUser RemoteSigned

If database tables are missing:
→ Run python init_db.py again

👨‍💻 Author

IAM Hiring – Local AI Matching System


---

# ✅ Why This Version Is Better

It:
- Assumes zero Python knowledge
- Explains what each command does
- Enforces correct execution order
- Explains repeat imports
- Makes the project self-contained

---

# 🎯 Where You Are Now

You have completed:

✔ CSV exploration  
✔ Normalized SQLite schema  
✔ Import pipeline  
✔ Deduplication  
✔ Git version control  

You are now officially past the “toy project” stage.

---

If you want next, we move to:

**Step 2 of the architecture: Building the embedding text generator.**

That’s where it becomes an AI system.

Ready?
