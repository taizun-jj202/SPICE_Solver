# def compare_files_unordered(file1_path, file2_path):
#     with open(file1_path, 'r') as f1, open(file2_path, 'r') as f2:
#         lines1 = set(line.strip() for line in f1 if line.strip())
#         lines2 = set(line.strip() for line in f2 if line.strip())

#     if lines1 == lines2:
#         print("✅ Files are the same (ignoring order).")
#     else:
#         print("❌ Files are different.")


def normalize_line(line):
    parts = line.strip().split()
    if len(parts) != 2:
        return line.strip()  # fallback for malformed lines

    identifier, value_str = parts
    try:
        value_float = float(value_str)
        value_normalized = f"{value_float:.6f}"  # fixed to 6 decimal places
    except ValueError:
        value_normalized = value_str  # fallback if not a float

    return f"{identifier} {value_normalized}"

def compare_files_unordered_normalized(file1_path, file2_path):
    with open(file1_path, 'r') as f1, open(file2_path, 'r') as f2:
        lines1 = set(normalize_line(line) for line in f1 if line.strip())
        lines2 = set(normalize_line(line) for line in f2 if line.strip())

    if lines1 == lines2:
        print("✅ Files are the same (ignoring order and formatting).")
    else:
        print("❌ Files are different.")

        only_in_file1 = lines1 - lines2
        only_in_file2 = lines2 - lines1

        if only_in_file1:
            print(f"\n📄 Lines only in {file1_path} ({len(only_in_file1)}):")
            for line in sorted(only_in_file1)[:10]:
                print("  ", line)

        if only_in_file2:
            print(f"\n📄 Lines only in {file2_path} ({len(only_in_file2)}):")
            for line in sorted(only_in_file2)[:10]:
                print("  ", line)

# Example usage
file1 = "/Users/taizunj/Documents/Masters_2024/ASU/Student_Docs/SEM2/EEE598_VLSI_Design_Automation/Mini_Project-2/PHASE-2/output.voltage"
file2 = "/Users/taizunj/Documents/Masters_2024/ASU/Student_Docs/SEM2/EEE598_VLSI_Design_Automation/Mini_Project-2/PHASE-2/test3.spout"
# compare_files_unordered(file1, file2)
# compare_files_unordered_verbose(file1, file2)
compare_files_unordered_normalized(file1, file2)
