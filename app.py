from flask import Flask, render_template, request
from transformers import pipeline

app = Flask(__name__)

# Load DistilGPT-2 for text generation
text_generator = pipeline("text-generation", model="distilgpt2")

def generate_content(topic):
    try:
        # Generate content using DistilGPT-2
        generated_text = text_generator(
            f"Write a detailed paragraph about {topic}:",
            max_length=150,  # Adjust the length of the generated text
            num_return_sequences=1,  # Number of responses to generate
            temperature=0.7,  # Controls randomness (0.0 = deterministic, 1.0 = creative)
            top_p=0.9,  # Controls diversity (lower values = more focused text)
            do_sample=True,  # Enables sampling for more diverse outputs
            repetition_penalty=1.5  # Penalizes repetition
        )
        # Remove repetitive sentences
        content = generated_text[0]['generated_text']
        sentences = content.split(". ")
        unique_sentences = []
        for sentence in sentences:
            if sentence not in unique_sentences:
                unique_sentences.append(sentence)
        return ". ".join(unique_sentences)
    except Exception as e:
        return f"Error generating content: {str(e)}"

@app.route('/', methods=['GET', 'POST'])
def index():
    content = ""
    if request.method == 'POST':
        topic = request.form['topic']
        content = generate_content(topic)
    return render_template('index.html', content=content)

if __name__ == '__main__':
    app.run(debug=True)