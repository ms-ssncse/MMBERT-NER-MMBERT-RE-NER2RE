def extract_text(input_file, output_file):
    imgidd=''
    with open(input_file, "r", encoding="utf-8") as f, open(output_file, "w", encoding="utf-8") as out_f:
        for line in f:
            line = line.strip()
            if line.startswith("IMGID:"):
                imgidd=line[6:]
                continue
            if line:
                parts = line.split()
                word = parts[0]
                out_f.write(f"{word} ")
            else:
                out_f.write(f"{imgidd}\n")
                out_f.write("\n")  

input_file = "Replace with input file path"
output_file = "Replace with output file path"  
extract_text(input_file, output_file)

print("Text extraction completed. Output saved to", output_file)