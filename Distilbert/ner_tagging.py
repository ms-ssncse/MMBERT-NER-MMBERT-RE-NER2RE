from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")

nlp = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="none")

def process_file(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as f, open(output_file, "w", encoding="utf-8") as out_f:
        for line in f:
            line = line.strip()
            if line: 
                parts = line.split()
                img_id = parts[-1] 
                sentence = " ".join(parts[:-1]) 
                
                out_f.write(f"IMGID:{img_id}\n")
                ner_results = nlp(sentence)
                
                merged_entities = []
                current_word = ""
                current_label = "O"
                
                for entity in ner_results:
                    word = entity["word"]
                    label = entity["entity"]
                    
                    if word.startswith("##"):  
                        current_word += word[2:]
                    else:
                        if current_word:
                            merged_entities.append({"word": current_word, "label": current_label})
                        current_word = word
                        current_label = label
                
                if current_word:
                    merged_entities.append({"word": current_word, "label": current_label})
                
                word_labels = {entity['word']: entity['label'] for entity in merged_entities}
                words = sentence.split()
                
                for word in words:
                    label = word_labels.get(word, "O")
                    out_f.write(f"{word} {label}\n")
                
                out_f.write("\n")  

input_file = "Replace with input file path"  
output_file = "Replace with output file path"  
process_file(input_file, output_file)

print("NER tagging completed. Output saved to", output_file)