import ast

def get_entity_type(relation, position):
    if relation == 'None':
        return "O"
    parts = relation.strip('/').split('/')
    if len(parts) > position:
        entity_type = parts[position]
        if entity_type == "per":
            return "PER"
        elif entity_type == "loc":
            return "LOC"
        elif entity_type == "org":
            return "ORG"
        elif entity_type == "misc":
            return "OTHER"
    return "O"

def process_sentences_fresh(input_filename, output_filename):
    instance_count = 0

    with open(input_filename, 'r', encoding="utf-8") as infile, open(output_filename, 'w', encoding="utf-8") as outfile:
        for line in infile:
            line = line.strip()
            if not line:
                continue  

            try:
                data_dict = ast.literal_eval(line)
            except (SyntaxError, ValueError):
                continue  

            token_list = data_dict.get('token', [])
            img_id = data_dict.get('img_id', '').replace(".jpg", "")
            head = data_dict.get('h', {})
            tail = data_dict.get('t', {})
            relation = data_dict.get('relation', 'None')

            head_pos = head.get('pos', [])
            tail_pos = tail.get('pos', [])

            head_entity_type = get_entity_type(relation, 0)
            tail_entity_type = get_entity_type(relation, 1)

            outfile.write(f"IMGID:{img_id}\n")
            instance_count += 1

            for i, token in enumerate(token_list):
                label = "O"

                if relation != 'None': 
                    if head_pos and head_pos[0] <= i < head_pos[1]:
                        label = f"B-{head_entity_type}" if i == head_pos[0] else f"I-{head_entity_type}"
                    elif tail_pos and tail_pos[0] <= i < tail_pos[1]:
                        label = f"B-{tail_entity_type}" if i == tail_pos[0] else f"I-{tail_entity_type}"

                outfile.write(f"{token}\t{label}\n")

            outfile.write("\n")

    print(f"Total sentences processed: {instance_count}")

input_file_path = "Replace with input file path"  
output_file_path_fresh = "Replace with output file path"
process_sentences_fresh(input_file_path, output_file_path_fresh)

print(f"Processed data has been written to {output_file_path_fresh}")