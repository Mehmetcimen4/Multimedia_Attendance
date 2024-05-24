import pandas as pd
from bs4 import BeautifulSoup
import sys

# Get the CSV file path from command-line arguments
csv_file_path = sys.argv[1] if len(sys.argv) > 1 else ''

if not csv_file_path:
    print("Error: No CSV file path provided.")
    sys.exit(1)

# Load the CSV file
df = pd.read_csv(csv_file_path)

# Extract Time, Participant, and Status columns
participants = df[['Time', 'Participant', 'Status']]

# Path to the HTML file
html_file_path = r'C:\Users\mehme\OneDrive\Masaüstü\Multimedia_Attendance\Multimedia_UI_Design.html'

# Read the HTML file
with open(html_file_path, 'r', encoding='utf-8') as file:
    soup = BeautifulSoup(file, 'html.parser')

# Find the table with id 'table' inside the div with id 'left_con'
table = soup.find('div', {'id': 'left_con'}).find('table', {'id': 'table'})

# Clear existing table contents
table_body = table.find('tbody')
if table_body:
    table_body.decompose()

# Create a new tbody element
table_body = soup.new_tag('tbody')

# Add the data rows
for _, row in participants.iterrows():
    data_row = soup.new_tag('tr')
    for cell in row:
        td = soup.new_tag('td')
        if pd.isna(cell):
            td.string = ''
        elif isinstance(cell, (int, float)):
            td.string = str(cell)
        else:
            td.string = cell
        data_row.append(td)
    table_body.append(data_row)

# Append the tbody to the table
table.append(table_body)

# Save the modified HTML back to the file
with open(html_file_path, 'w', encoding='utf-8') as file:
    file.write(str(soup))

print("HTML file has been updated successfully.")
