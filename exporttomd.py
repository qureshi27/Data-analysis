import pandas as pd

# Path to your CSV file (root directory)
csv_file = "batch_details.csv"   # replace with your filename

# Read CSV
df = pd.read_csv(csv_file)

# Prepare Markdown content
md_content = "# CSV File Preview\n\n"

# Add column names
md_content += "## Columns\n\n"
md_content += ", ".join(df.columns) + "\n\n"

# Add first 5 rows
md_content += "## First 5 Rows\n\n"
md_content += df.head().to_markdown(index=False) + "\n"

# Export to Markdown file
output_file = "csv_preview.md"
with open(output_file, "w", encoding="utf-8") as f:
    f.write(md_content)

print(f"Markdown file saved as {output_file}")
