import fitz

pdf_path = 'TaiLieu/Figures/global-figures.pdf'
pdf = fitz.open(pdf_path)
page = pdf[80] # Page 81 (0-indexed)

print("Text on page 81:")
blocks = page.get_text("blocks")
for b in blocks:
    print(b)
    
# Let's search for "Figure 11"
print("\nSearching for 'Figure 11':")
results = page.search_for("Figure 11")
print(results)
pdf.close()
