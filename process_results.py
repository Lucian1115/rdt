import os
import argparse
import statistics

def calculate_and_write(f_out, config, scores):
    if not scores:
        return
    mean_val = statistics.mean(scores)
    std_val = statistics.stdev(scores) if len(scores) > 1 else 0.0
    f_out.write(f"{config}: {mean_val:.2f} ({std_val:.2f})\n")

def process_directory(folder_path):
    if not os.path.isdir(folder_path):
        print(f"Error: Directory '{folder_path}' not found.")
        return

    files_found = 0

    for filename in os.listdir(folder_path):
        if filename.startswith("test_") and filename.endswith(".txt") and "_processed" not in filename:
            output_filename = filename.replace(".txt", "_processed.txt")
            files_found += 1
            input_path = os.path.join(folder_path, filename)
            output_path = os.path.join(folder_path, output_filename)

            print(f"Processing {filename}...")

            try:
                with open(input_path, 'r', encoding='utf-8') as f_in, open(output_path, 'w', encoding='utf-8') as f_out:
                    current_config = None
                    current_scores = []

                    for line in f_in:
                        line = line.strip()
                        if not line:
                            continue
                        
                        try:
                            identifier, score_str = line.split(': ')
                            config = identifier.rsplit('_', 1)[0]
                            score = float(score_str)

                            if config != current_config:
                                if current_config is not None:
                                    calculate_and_write(f_out, current_config, current_scores)
                                current_config = config
                                current_scores = [score]
                            else:
                                current_scores.append(score)
                        except ValueError:
                            continue

                    if current_config is not None:
                        calculate_and_write(f_out, current_config, current_scores)
                
                print(f"Saved to {output_filename}")

            except Exception as e:
                print(f"Failed to process {filename}: {e}")
    
    if files_found == 0:
        print("No matching files found.")
    else:
        print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", help="Path to the folder containing txt files")
    args = parser.parse_args()
    process_directory(args.folder)