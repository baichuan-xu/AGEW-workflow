import dashscope
import os
import time

# --- Configuration ---
# The base path where the Lxxxx subfolders are located.
# os.getcwd() gets the current working directory where the script is run.
BASE_PATH = os.getcwd()
SUBFOLDER_PREFIX = "L"
NUM_SUBFOLDERS = 3356
YS_PREFIX = "YS"
INPUT_IMAGE_NAME = "in.png"
# The full path to the markdown file containing the LLM prompt.
PROMPT_FILE_PATH = r".\stress-strain\prompt\figure_split.md"
OUTPUT_CSV_NAME = "figure_split.csv"

def read_text_file(file_path):
    """
    Reads and returns the content of a text file.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as text_file:
            return text_file.read()
    except FileNotFoundError:
        print(f"Error: Prompt file not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None

def write_to_csv(file_path, content):
    """
    Writes the given content directly to a CSV file.
    Assumes the content is already in a CSV-compatible string format.
    """
    try:
        # Ensure content is a string before writing
        if not isinstance(content, str):
            raise TypeError(f"Content to be written must be a string, not {type(content)}")
            
        with open(file_path, "w", encoding="utf-8") as csv_file:
            # The LLM output is written directly to the file.
            # Ensure your prompt asks the LLM to format the text as a CSV string.
            csv_file.write(content)
        print(f"Successfully wrote output to {file_path}")
    except Exception as e:
        print(f"Error writing to file {file_path}: {e}")

def process_image_with_llm(image_path, prompt_text, output_csv_path, api_key):
    """
    Sends a single image and a text prompt to the DashScope LLM and saves the response.
    """
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return False

    # Format the local image path for the API call.
    local_image_path_for_api = f"file://{os.path.abspath(image_path)}"
    messages = [
        {
            "role": "system",
            "content": "You are a highly accurate assistant for scientific figure analysis. Your task is to analyze the provided image and extract information precisely as instructed."
        },
        {
            "role": "user",
            "content": [
                {"image": local_image_path_for_api},
                {"text": prompt_text}
            ]
        }
    ]

    print(f"Sending request to DashScope LLM for {image_path}...")
    try:
        response = dashscope.MultiModalConversation.call(
            api_key=api_key,
            model='qwen-vl-max',
            messages=messages,
            temperature=0.0, # Set to 0.0 for deterministic and consistent output
        )

        if response.status_code == 200 and response.output and response.output.choices:
            # BUG FIX: The API can return content as a list (e.g., [{'text': '...'}]) or a direct string.
            # This logic handles both cases to ensure we always have a string.
            llm_output = response.output.choices[0].message.content
            
            final_text_output = ""
            if isinstance(llm_output, list) and llm_output:
                # If the output is a list, extract the text from the first element.
                content_part = llm_output[0]
                if isinstance(content_part, dict):
                    final_text_output = content_part.get("text", "")
                else:
                    final_text_output = str(content_part)
            elif isinstance(llm_output, str):
                final_text_output = llm_output
            else:
                # Fallback for any other unexpected type
                final_text_output = str(llm_output)

            write_to_csv(output_csv_path, final_text_output)

            if response.usage:
                print("Token Usage:")
                print(f"  Input Tokens: {response.usage.get('input_tokens', 'N/A')}")
                print(f"  Output Tokens: {response.usage.get('output_tokens', 'N/A')}")
            return True
        else:
            print(f"Error from API: Status Code {response.status_code}")
            print(f"Response details: {response}")
            return False
    except Exception as e:
        print(f"An exception occurred during the API call for {image_path}: {e}")
        return False

def main():
    """
    Main function to iterate through directories and process images.
    """
    # Read the prompt file once at the beginning to avoid re-reading it in the loop.
    prompt_text = read_text_file(PROMPT_FILE_PATH)
    if prompt_text is None:
        print("Aborting script because the prompt file could not be read.")
        return

    # Retrieve API key from environment variables.
    # Replace "YOUR_API_KEY_HERE" with your actual key if not using environment variables.
    api_key = os.getenv('DASHSCOPE_API_KEY')
    if not api_key:
        print("Warning: Using hardcoded API key. Consider setting DASHSCOPE_API_KEY environment variable.")
    if not api_key:
        print("Error: No API key provided.")
        return

    print(f"Starting batch processing in directory: {BASE_PATH}")
    for i in range(1, NUM_SUBFOLDERS + 1):
        subfolder_name = f"{SUBFOLDER_PREFIX}{i:04d}"
        l_subfolder_path = os.path.join(BASE_PATH, subfolder_name)

        if not os.path.isdir(l_subfolder_path):
            # This check is useful if some numbers in the sequence are missing.
            print(f"Directory {l_subfolder_path} not found. Skipping.")
            continue

        print(f"\n--- Checking {l_subfolder_path} ---")

        # Find the subfolder starting with "YS"
        ys_subfolder_path = None
        for item in os.listdir(l_subfolder_path):
            if item.startswith(YS_PREFIX) and os.path.isdir(os.path.join(l_subfolder_path, item)):
                ys_subfolder_path = os.path.join(l_subfolder_path, item)
                break # Found the YS folder, no need to check further in this L-folder.

        if not ys_subfolder_path:
            print(f"No subfolder starting with '{YS_PREFIX}' found in {l_subfolder_path}. Skipping.")
            continue

        # Define the paths for the input image and output CSV file.
        image_path = os.path.join(ys_subfolder_path, INPUT_IMAGE_NAME)
        output_csv_path = os.path.join(ys_subfolder_path, OUTPUT_CSV_NAME)

        if not os.path.exists(image_path):
            print(f"Input image '{INPUT_IMAGE_NAME}' not found in {ys_subfolder_path}. Skipping.")
            continue

        print(f"Processing image in: {ys_subfolder_path}")
        process_image_with_llm(image_path, prompt_text, output_csv_path, api_key)

        # Optional: Add a small delay to avoid hitting API rate limits if you have many requests.
        # time.sleep(1)

    print("\n--- Batch processing complete. ---")

if __name__ == "__main__":
    main()

