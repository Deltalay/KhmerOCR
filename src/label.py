import os
from lxml import etree

def process_xml_folder(input_folder, output_folder):
    """
    Processes all XML files in an input folder and saves the modified
    versions to an output folder.

    Args:
        input_folder (str): The path to the folder containing original XML files.
        output_folder (str): The path to the folder where modified files will be saved.
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    print(f"Output will be saved to: {output_folder}")
    for filename in os.listdir(input_folder):
        if filename.lower().endswith('.xml'):
            input_file_path = os.path.join(input_folder, filename)
            output_file_path = os.path.join(output_folder, filename)
            
            print(f"Processing '{filename}'...")

            try:
                # Parse the original XML file using lxml
                tree = etree.parse(input_file_path)
                root = tree.getroot()

                # Create a new root for the output XML
                new_root = etree.Element(root.tag)

                # Copy metadata elements (image, width, height)
                for child in root:
                    if child.tag != 'paragraph':
                        new_root.append(child)

                # Create the new paragraph element
                new_paragraph = etree.SubElement(new_root, 'paragraph')

                # Iterate through each <line> in the original XML
                for line in root.xpath('.//line'):
                    words = line.xpath('./word')
                    if not words:
                        continue

                    # 1. Combine all word texts without spaces
                    line_text = "".join(word.find('text').text for word in words)

                    # 2. Calculate the merged bounding box
                    x1s = [int(w.find('bbox').get('x1')) for w in words]
                    y1s = [int(w.find('bbox').get('y1')) for w in words]
                    x2s = [int(w.find('bbox').get('x2')) for w in words]
                    y2s = [int(w.find('bbox').get('y2')) for w in words]
                    
                    line_bbox_attrs = {
                        "x1": str(min(x1s)), "y1": str(min(y1s)),
                        "x2": str(max(x2s)), "y2": str(max(y2s))
                    }

                    # Create the new <line> element
                    new_line = etree.SubElement(new_paragraph, 'line', attrib={'id': line.get('id')})
                    etree.SubElement(new_line, 'text').text = line_text
                    etree.SubElement(new_line, 'bbox', attrib=line_bbox_attrs)

                # Write the new, modified XML tree to the output file
                new_tree = etree.ElementTree(new_root)
                new_tree.write(output_file_path, pretty_print=True, encoding='utf-8')

            except Exception as e:
                print(f"  Error processing {filename}: {e}")

    print("\nProcessing complete.")

if __name__ == "__main__":
    INPUT_DIRECTORY = "synthetic_xml_labels"
    OUTPUT_DIRECTORY = "fix_labels"

    # Run the processing function
    process_xml_folder(INPUT_DIRECTORY, OUTPUT_DIRECTORY)