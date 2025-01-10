import os
import xml.etree.ElementTree as ET
import pandas as pd

def parse_xml_to_dict(file_path):
    """
    Parses an XML file and returns a dictionary representing its data.
    Handles nested XML elements by flattening them with hierarchical keys.
    """
    def recursive_parse(element, parent_key=""):
        data = {}
        for child in element:
            key = f"{parent_key}.{child.tag}" if parent_key else child.tag
            if len(child):
                # Recursively parse nested elements
                data.update(recursive_parse(child, key))
            else:
                # Add text content of the element
                data[key] = child.text
        return data

    tree = ET.parse(file_path)
    root = tree.getroot()
    return recursive_parse(root)

def convert_xml_folder_to_csv(folder_path, output_csv):
    """
    Reads all XML files in the specified folder, parses their content,
    and writes them into a single CSV file.
    """
    all_data = []

    # Check if folder exists
    if not os.path.isdir(folder_path):
        print(f"Error: The folder '{folder_path}' does not exist.")
        return

    # Iterate over all XML files in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.xml'):
            file_path = os.path.join(folder_path, file_name)
            try:
                data = parse_xml_to_dict(file_path)
                all_data.append(data)
            except Exception as e:
                print(f"Error parsing {file_name}: {e}")

    # Convert list of dictionaries to a DataFrame and save as CSV
    if all_data:
        df = pd.DataFrame(all_data)
        df.to_csv(output_csv, index=False)
        print(f"CSV file saved as {output_csv}")
    else:
        print("No valid XML files found.")

if __name__ == "__main__":
    # Specify the folder containing XML files and the output CSV file name
    folder_path = input("Enter the folder path containing XML files: ")
    output_csv = input("Enter the output CSV file name (e.g., output.csv): ")

    # Run the conversion
    convert_xml_folder_to_csv(folder_path, output_csv)
