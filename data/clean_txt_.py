from pathlib import Path
from pythainlp.tokenize import word_tokenize

input_path = Path("data/%s.txt"%input("Enter the name of the input file (without extension): "))
output_path = Path("data/eng-%s.txt"% input_path.stem)

print('\n')
with open(input_path, encoding='utf-8') as f_in, open(output_path, encoding='utf-8', mode='w') as f_out:

    print(f"Processing {input_path} and writing to {output_path}...")
    for line in list(f_in):
        parts = line.strip().split("\t")
        # delete the last part if there are more than 2 parts
        parts = parts[:2]

        if input_path.name == "tha.txt":
            # tokenize the Thai part
            parts[1] = ' '.join(word_tokenize(parts[1], engine="newmm", keep_whitespace=False))

        f_out.write(parts[0] + "\t" + parts[1] + "\n")

print('\n')
print('Done!')
