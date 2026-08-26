"""Replace inline math delimiters $...$ with \(...\) without affecting $$...$$"""

import re

file_path = r'd:\广东电网\深圳电网\EditedDistOPF_Python_Ver\README.md'

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Match single $...$ but NOT $$...$$
# (?<!\$)\$(?!\$)  = a $ that is not preceded or followed by another $
# (?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)  = $content$ where both $ are single
result = re.sub(
    r'(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)',
    r'\\(\1\\)',
    content,
    flags=re.DOTALL
)

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(result)

print("Replacement complete.")