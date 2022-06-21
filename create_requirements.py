import re
import os

folder = '.'

files = [f for f in os.listdir(folder) if f.endswith(".py")]

imports = []
for file in files:
    with open(os.path.join(folder, file), mode="r") as f:
        lines = f.read()
        #result = re.findall(r"(?<!from)import (\w+)[\n.]|from\s+(\w+)\s+import|(?<!from)import (\w+)[\as]", lines)
        result = re.findall(r"(?<!from)import (\w+)[\n.]|from\s+(\w+)\s+import|import\s+(\w+)\s+as", lines)
        for imp in result:
            for i in imp:
                if len(i):
                    if i not in imports:
                        imports.append(i)
                        
list_to_remove = ['os','re']
final_list = list(set(imports) - set(list_to_remove))
print(final_list)
#print(len(final_list))
file = open("requirements.txt", "w+")
for imported in final_list:
    content = str(imported)
    file.write(f'{content}\n')

file.close()
