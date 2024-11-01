import requests
import fitz  # PyMuPDF
import google.generativeai as genai
import textwrap
import csv
import json
import os
import time

# Step 1: Set up API key securely
api_key = "Gemini_API_Key"
if not api_key:
    raise ValueError("API key for Google Generative AI not set in environment variables.")
genai.configure(api_key=api_key)

# Step 2: Download and extract text from the PDF
pdf_url = 'https://ncert.nic.in/textbook/pdf/lebo101.pdf'
response = requests.get(pdf_url)

if response.status_code == 200:
    with open('lebo101.pdf', 'wb') as f:
        f.write(response.content)
    print("PDF downloaded successfully.")
else:
    print("Failed to download PDF.")

# Step 3: Extract text from the PDF
doc = fitz.open('lebo101.pdf')
full_text = ""

for page in doc:
    full_text += page.get_text()

doc.close()

# Step 4: Split the text into topics
topics = full_text.split('\n\n')  # Adjust as needed

# Step 5: Define Bloom's Taxonomy levels with improved prompts
bloom_prompts = {
    "Remembering": "Generate a question that tests recall of the fundamental concepts discussed in the text. Include key terms and definitions. Provide a clear answer that lists these terms with their definitions.",
    "Understanding": "Formulate a question that requires an explanation of the main ideas from the text. The answer should summarize the concepts in a way that demonstrates comprehension.",
    "Applying": "Create a question that asks for a real-world application of the concepts presented in the text. The answer should describe a scenario where these concepts can be effectively applied.",
    "Analyzing": "Devise a question that encourages analysis of the relationships between different concepts in the text. The answer should include a comparison or contrast of these concepts.",
    "Evaluating": "Construct a question that asks for an evaluation of the effectiveness of the ideas presented in the text. The answer should include criteria for assessment and a reasoned judgment based on these criteria.",
    "Creating": "Formulate a question that invites the generation of new ideas or solutions based on the information in the text. The answer should propose an innovative approach or concept."
}

# Configure the Generative AI model
model = genai.GenerativeModel(
    model_name='gemini-1.5-flash'
)

# Step 6: Create a CSV file to store the results
csv_filename = 'bloom_questions_updated_prompt.csv'

try:
    with open(csv_filename, mode='w', newline='', encoding='utf-8') as csv_file:
        fieldnames = ['summarized_topic', 'question', 'answer', 'bloom_level']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        print(f"CSV file '{csv_filename}' created successfully.")

        # Step 7: Generate questions and answers for each topic
        for topic in topics:
            # Summarize the topic
            summarized_topic_prompt = textwrap.dedent(f"""\
                Summarize the following text in a few sentences:
                {topic}
            """)
            try:
                summary_response = model.generate_content(
                    summarized_topic_prompt,
                    generation_config={'response_mime_type': 'application/json'}
                )

                if hasattr(summary_response, 'candidates') and summary_response.candidates:
                    summarized_topic = summary_response.candidates[0].content.parts[0].text.strip()

                    # Remove "summary": from the summarized_topic if it exists
                    if summarized_topic.startswith('{"summary":'):
                        summarized_topic = summarized_topic.replace('{"summary": "', '').rstrip('"}')
                else:
                    summarized_topic = "No summary generated"

                # Step 8: Generate questions and answers for each Bloom's level
                for level, prompt in bloom_prompts.items():
                    for i in range(5):  # Loop to generate 5 different questions and answers
                        llm_prompt = textwrap.dedent(f"""\
                            Based on Bloom's Taxonomy, generate a question and answer for the level "{level}".
                            
                            Instruction: {prompt}
                            
                            Text:
                            {topic}
                            
                            Important: Only return a single JSON object with "question" and "answer" keys.
                            Example JSON:
                            {{
                                "question": "Generated question here",
                                "answer": "Generated answer here"
                            }}
                        """)

                        # Call the model to generate content
                        try:
                            response = model.generate_content(
                                llm_prompt,
                                generation_config={'response_mime_type': 'application/json'}
                            )

                            if hasattr(response, 'candidates') and response.candidates:
                                generated_text = response.candidates[0].content.parts[0].text
                                result_data = json.loads(generated_text)

                                # Write to CSV
                                writer.writerow({
                                    'summarized_topic': summarized_topic,
                                    'question': result_data.get('question', 'No question generated'),
                                    'answer': result_data.get('answer', 'No answer generated'),
                                    'bloom_level': level
                                })

                                print(f"{level} Level - Question {i + 1} and Answer saved to CSV for topic summarized as: '{summarized_topic}'.\n")
                            else:
                                print(f"No candidates found in the response for level {level}.")

                        except json.JSONDecodeError:
                            print(f"Error decoding JSON for level {level}, Question {i + 1}. Response: {generated_text}")
                        except Exception as e:
                            print(f"Error generating content for {level}, Question {i + 1}: {str(e)}")

                        # Optional: Sleep to avoid rate limiting
                        time.sleep(1)  # Adjust the sleep time as needed

            except Exception as e:
                print(f"Error summarizing topic: {str(e)}")

except Exception as e:
    print(f"Failed to create CSV file: {str(e)}")
