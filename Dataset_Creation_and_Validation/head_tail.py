import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import re
import os

# Load CLIP model and processor
model_name = "openai/clip-vit-base-patch32"
processor = CLIPProcessor.from_pretrained(model_name)
model = CLIPModel.from_pretrained(model_name)

# Define input and output file paths
text_file_path = "/content/extracted_sentences_with_img_id.txt"  # Update with actual text file path
image_folder = ""  # Update with actual image folder path
output_file_path = ""  # Output file

# Function to extract named entities
def extract_named_entities(sentence):
    named_entities = re.findall(r"\b(?:[A-Z][a-z]+(?:\s[A-Z][a-z]+)*)\b", sentence)

    # Ensure at least two words are extracted
    if len(named_entities) < 2:
        words = sentence.split()
        additional_words = [word for word in words if word.istitle() and word not in named_entities]
        named_entities.extend(additional_words[:2 - len(named_entities)])

    return named_entities if len(named_entities) >= 2 else ["Unknown", "Unknown"]

# Process each line in the text file
with open(text_file_path, "r", encoding="utf-8") as infile, open(output_file_path, "w", encoding="utf-8") as outfile:
    for line in infile:
        line = line.strip()
        if not line:
            continue  # Skip empty lines

        # Extract last token as IMGID
        parts = line.split()
        img_id = parts[-1]
        sentence = " ".join(parts[:-1])
        image_path = os.path.join(image_folder, f"{img_id}")  # Adjust format if needed

        # Ensure image exists
        if not os.path.exists(image_path):
            print(f"Warning: Image {image_path} not found. Skipping...")
            continue

        image = Image.open(image_path).convert("RGB")
        named_entities = extract_named_entities(sentence)

        # Process inputs
        inputs = processor(text=named_entities, images=image, return_tensors="pt", padding=True)

        # Compute similarity scores
        with torch.no_grad():
            outputs = model(**inputs)
            text_features = outputs.text_embeds  # Text embeddings
            image_features = outputs.image_embeds  # Image embeddings
            similarity = (text_features @ image_features.T).squeeze(1)  # Compute similarity scores

        # Find the most relevant (head) and second most relevant (tail) entity
        top_indices = similarity.argsort(descending=True)[:2]
        top_entities = [named_entities[idx] for idx in top_indices]
        print(top_entities)

        # Write results to output file
        outfile.write(f"IMGID:{img_id}\n")
        outfile.write(f"Head Entity: {top_entities[0]}\n")
        outfile.write(f"Tail Entity: {top_entities[1]}\n\n")

print(f"Processing complete. Results saved to {output_file_path}")