from flask import Flask, render_template, request, jsonify
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.tokenize.treebank import TreebankWordDetokenizer

nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)

def extractive_summarization(text, num_sentences=3):
    sentences = sent_tokenize(text)
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word.lower() not in stop_words]
    freq_dist = FreqDist(filtered_words)
    sentence_scores = {}
    for sentence in sentences:
        for word, freq in freq_dist.items():
            if word.lower() in sentence.lower():
                sentence_scores[sentence] = sentence_scores.get(sentence, 0) + freq
    top_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:num_sentences]
    summary = TreebankWordDetokenizer().detokenize(top_sentences)
    return summary

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    if request.method == 'POST':
        input_text = request.form['text']
        summary = extractive_summarization(input_text)
        return jsonify({'input_text': input_text, 'summary': summary})

if __name__ == '__main__':
    app.run(debug=True)
