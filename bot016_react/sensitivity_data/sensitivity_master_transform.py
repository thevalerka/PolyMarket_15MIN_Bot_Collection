import json

def transform_sensitivity_data(input_file, output_file):
    # Load the JSON data from the input file
    with open(input_file, 'r') as f:
        data = json.load(f)

    # Iterate over each bin in the 'bins' dictionary
    for key, value in data['bins'].items():
        # Remove 'put_sensitivity_raw' if it exists
        if 'put_sensitivity_raw' in value:
            del value['put_sensitivity_raw']
        # Remove 'call_sensitivity_raw' if it exists
        if 'call_sensitivity_raw' in value:
            del value['call_sensitivity_raw']

    # Write the modified data to the output file
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)

# Example usage:
transform_sensitivity_data('sensitivity_master.json', 'sensitivity_transformed.json')
