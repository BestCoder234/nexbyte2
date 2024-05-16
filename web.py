from flask import Flask, request, render_template_string
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import re
import nltk
nltk.download('punkt')
from nltk import sent_tokenize

class Adequacy:
    def __init__(self, model_tag='prithivida/parrot_adequacy_model'):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.adequacy_model = AutoModelForSequenceClassification.from_pretrained(model_tag)
        self.tokenizer = AutoTokenizer.from_pretrained(model_tag)

    def filter(self, input_phrase, para_phrases, adequacy_threshold, device="cpu"):
        top_adequacy_phrases = []
        for para_phrase in para_phrases:
            x = self.tokenizer(input_phrase, para_phrase, return_tensors='pt', max_length=128, truncation=True)
            x = x.to(device)
            self.adequacy_model = self.adequacy_model.to(device)
            logits = self.adequacy_model(**x).logits
            probs = logits.softmax(dim=1)
            prob_label_is_true = probs[:, 1]
            adequacy_score = prob_label_is_true.item()
            if adequacy_score >= adequacy_threshold:
                top_adequacy_phrases.append(para_phrase)
        return top_adequacy_phrases

class Fluency:
    def __init__(self, model_tag='prithivida/parrot_fluency_model'):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.fluency_model = AutoModelForSequenceClassification.from_pretrained(model_tag, num_labels=2)
        self.fluency_tokenizer = AutoTokenizer.from_pretrained(model_tag)

    def filter(self, para_phrases, fluency_threshold, device="cpu"):
        import numpy as np
        from scipy.special import softmax
        self.fluency_model = self.fluency_model.to(device)
        top_fluent_phrases = []
        for para_phrase in para_phrases:
            input_ids = self.fluency_tokenizer("Sentence: " + para_phrase, return_tensors='pt', truncation=True)
            input_ids = input_ids.to(device)
            prediction = self.fluency_model(**input_ids)
            scores = prediction[0][0].detach().cpu().numpy()
            scores = softmax(scores)
            fluency_score = scores[1]  # LABEL_0 = Bad Fluency, LABEL_1 = Good Fluency
            if fluency_score >= fluency_threshold:
                top_fluent_phrases.append(para_phrase)
        return top_fluent_phrases

class Diversity:
    def __init__(self, model_tag='paraphrase-distilroberta-base-v2'):
        from sentence_transformers import SentenceTransformer
        self.diversity_model = SentenceTransformer(model_tag)

    def rank(self, input_phrase, para_phrases, diversity_ranker='levenshtein'):
        if diversity_ranker == "levenshtein":
            return self.levenshtein_ranker(input_phrase, para_phrases)
        elif diversity_ranker == "euclidean":
            return self.euclidean_ranker(input_phrase, para_phrases)
        elif diversity_ranker == "diff":
            return self.diff_ranker(input_phrase, para_phrases)

    def euclidean_ranker(self, input_phrase, para_phrases):
        import pandas as pd
        from sklearn_pandas import DataFrameMapper
        from sklearn.preprocessing import MinMaxScaler
        from scipy import spatial

        diversity_scores = {}
        outputs = []
        input_enc = self.diversity_model.encode(input_phrase.lower())
        for para_phrase in para_phrases:
            paraphrase_enc = self.diversity_model.encode(para_phrase.lower())
            euclidean_distance = spatial.distance.euclidean(input_enc, paraphrase_enc)
            outputs.append((para_phrase, euclidean_distance))
        df = pd.DataFrame(outputs, columns=['paraphrase', 'scores'])
        fields = []
        for col in df.columns:
            if col == "scores":
                tup = ([col], MinMaxScaler())
            else:
                tup = ([col], None)
            fields.append(tup)

        mapper = DataFrameMapper(fields, df_out=True)
        for index, row in mapper.fit_transform(df.copy()).iterrows():
            diversity_scores[row['paraphrase']] = row['scores']
        return diversity_scores

    def levenshtein_ranker(self, input_phrase, para_phrases):
        import Levenshtein
        diversity_scores = {}
        for para_phrase in para_phrases:
            distance = Levenshtein.distance(input_phrase.lower(), para_phrase)
            diversity_scores[para_phrase] = distance
        return diversity_scores

    def diff_ranker(self, input_phrase, para_phrases):
        import difflib
        differ = difflib.Differ()
        diversity_scores = {}
        for para_phrase in para_phrases:
            diff = differ.compare(input_phrase.split(), para_phrase.split())
            count = 0
            for d in diff:
                if "+" in d or "-" in d:
                    count += 1
            diversity_scores[para_phrase] = count
        return diversity_scores

class Parrot:
    def __init__(self, model_tag="prithivida/parrot_paraphraser_on_T5", use_gpu=False):
        self.tokenizer = AutoTokenizer.from_pretrained(model_tag, use_auth_token=False)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_tag, use_auth_token=False)
        self.adequacy_score = Adequacy()
        self.fluency_score = Fluency()
        self.diversity_score = Diversity()
        self.device = "cuda:0" if use_gpu and torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def _clean_text(self, text):
        """Utility function to clean text by removing unwanted characters"""
        return re.sub('[^a-zA-Z0-9 \?\'\-\/\:\.]', '', text).lower()

    def _generate_paraphrases(self, input_phrase, max_length, max_return_phrases, do_diverse):
        """Generates paraphrases for a given input phrase"""
        input_phrase = self._clean_text(input_phrase)
        input_ids = self.tokenizer.encode("paraphrase: " + input_phrase, return_tensors='pt').to(self.device)
        if do_diverse:
            for n in range(2, 9):
                if max_return_phrases % n == 0:
                    break
            preds = self.model.generate(
                input_ids,
                do_sample=False,
                max_length=max_length,
                num_beams=max_return_phrases,
                num_beam_groups=n,
                diversity_penalty=2.0,
                early_stopping=True,
                num_return_sequences=max_return_phrases)
        else:
            preds = self.model.generate(
                input_ids,
                do_sample=True,
                max_length=max_length,
                top_k=50,
                top_p=0.95,
                early_stopping=True,
                num_return_sequences=max_return_phrases)

        paraphrases = set(self.tokenizer.decode(pred, skip_special_tokens=True) for pred in preds)
        return self._clean_paraphrases(paraphrases)

    def _clean_paraphrases(self, paraphrases):
        """Utility function to clean generated paraphrases"""
        return {self._clean_text(phrase) for phrase in paraphrases}

    def _filter_and_rank_paraphrases(self, input_phrase, paraphrases, adequacy_threshold, fluency_threshold, diversity_ranker):
        """Filters and ranks paraphrases based on adequacy, fluency, and diversity"""
        adequacy_filtered_phrases = self.adequacy_score.filter(input_phrase, paraphrases, adequacy_threshold, self.device)
        if not adequacy_filtered_phrases:
            return []

        fluency_filtered_phrases = self.fluency_score.filter(adequacy_filtered_phrases, fluency_threshold, self.device)
        if not fluency_filtered_phrases:
            return []

        diversity_scored_phrases = self.diversity_score.rank(input_phrase, fluency_filtered_phrases, diversity_ranker)
        ranked_phrases = sorted(diversity_scored_phrases.items(), key=lambda x: x[1], reverse=True)

        return ranked_phrases

    def paraphrase_sentence(self, sentence, diversity_ranker="levenshtein", do_diverse=False, max_length=512, adequacy_threshold=0.90, fluency_threshold=0.90, max_return_phrases=10):
        """Paraphrases a single sentence"""
        paraphrases = self._generate_paraphrases(sentence, max_length, max_return_phrases, do_diverse)
        filtered_and_ranked = self._filter_and_rank_paraphrases(sentence, paraphrases, adequacy_threshold, fluency_threshold, diversity_ranker)
        return filtered_and_ranked

    def paraphrase_essay(self, essay, diversity_ranker="levenshtein", do_diverse=False, max_length=512, adequacy_threshold=0.90, fluency_threshold=0.90, max_return_phrases=10):
        """Paraphrases an entire essay sentence by sentence"""
        sentences = sent_tokenize(essay)
        paraphrased_sentences = []

        for sentence in sentences:
            paraphrased = self.paraphrase_sentence(sentence, diversity_ranker, do_diverse, max_length, adequacy_threshold, fluency_threshold, max_return_phrases)
            if paraphrased:
                paraphrased_sentences.append(paraphrased[0][0])

        return ' '.join(paraphrased_sentences)

# Flask app setup
app = Flask(__name__)
parrot_instance = Parrot(use_gpu=False)

@app.route('/', methods=['GET', 'POST'])
def home():
    paraphrased_essay = ""
    diversity_ranker = request.form.get('diversity_ranker', 'levenshtein')
    fluency_threshold = float(request.form.get('fluency_threshold', 0.9))

    if request.method == 'POST':
        essay = request.form.get('essay')
        if essay:
            paraphrased_essay = parrot_instance.paraphrase_essay(
                essay,
                diversity_ranker=diversity_ranker,
                fluency_threshold=fluency_threshold
            )

    return render_template_string("""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Paraphrase Your Essay</title>
        <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.3/css/all.min.css" rel="stylesheet">
        <style>
            body {
                font-family: Arial, sans-serif; margin: 0; background-color: #f4f4f4;
                color: #333; padding: 0;
            }
            .container {
                max-width: 900px; margin: auto; padding: 40px; background-color: #fff;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1); border-radius: 8px; margin-top: 50px;
                text-align: center;
            }
            header {
                margin-bottom: 20px;
            }
            header img {
                height: 80px; margin-right: 10px;
            }
            h1 {
                text-align: center; color: #333;
                font-size: 2.5em;
                margin-bottom: 10px;
            }
            .subtitle {
                color: #666;
                font-size: 1.2em;
                margin-bottom: 40px;
            }
            textarea {
                width: 100%; height: 200px; margin-bottom: 20px; padding: 15px;
                border: 1px solid #ddd; border-radius: 4px; font-family: Arial, sans-serif;
                resize: vertical; font-size: 1em;
            }
            input[type='submit'] {
                padding: 15px 30px; background-color: #007BFF; color: white;
                border: none; border-radius: 4px; cursor: pointer;
                font-size: 1.2em;
                transition: background-color 0.3s;
            }
            input[type='submit']:hover {
                background-color: #0056b3;
            }
            select, input[type='range'], label {
                margin-bottom: 10px; padding: 10px; border-radius: 4px;
                display: block;
                width: 100%;
                font-size: 1em;
            }
            .settings {
                display: flex; flex-direction: column; align-items: center; margin-bottom: 20px;
            }
            .settings div {
                width: 100%; max-width: 400px; margin-bottom: 20px;
            }
            .slider-container {
                text-align: left;
                width: 100%; max-width: 400px;
            }
            .slider-label {
                font-size: 1em;
                margin-bottom: 5px;
            }
            .slider {
                width: 100%;
            }
            h2, pre {
                margin-top: 20px; color: #333;
            }
            pre {
                background-color: #f9f9f9; padding: 15px; border: 1px solid #ddd; border-radius: 4px;
                white-space: pre-wrap; word-wrap: break-word;
                text-align: left;
                font-size: 1em;
            }
            .footer {
                text-align: center; margin-top: 30px; font-size: 14px; color: #777;
            }
            .cta-buttons {
                display: flex; justify-content: space-around; margin-top: 20px;
            }
            .cta-buttons a {
                padding: 10px 20px; background-color: #28a745; color: white;
                border: none; border-radius: 4px; cursor: pointer;
                text-decoration: none;
                transition: background-color 0.3s;
                font-size: 1.2em;
            }
            .cta-buttons a:hover {
                background-color: #218838;
            }
            .cta-buttons a.secondary {
                background-color: #ffc107;
            }
            .cta-buttons a.secondary:hover {
                background-color: #e0a800;
            }
            .description {
                text-align: left;
                margin-top: 50px;
            }
            .description h3 {
                margin-top: 30px;
                color: #007BFF;
            }
            .description p {
                margin-bottom: 20px;
                line-height: 1.6;
            }
            .description ul {
                list-style: none;
                padding-left: 0;
            }
            .description ul li {
                margin-bottom: 10px;
                line-height: 1.6;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <header>
                <img src="/static/logo.png" alt="NexByte Logo">
                <h1>Paraphrase Your Essay</h1>
                <p class="subtitle">Humanize AI Text with the Best Paraphrasing Tool</p>
            </header>
            <form method="post">
                <textarea name="essay" rows="10" cols="50" placeholder="Paste your essay here..."></textarea>
                <div class="settings">
                    <div>
                        <label for="diversity_ranker">Diversity Ranking Method:</label>
                        <select name="diversity_ranker">
                            <option value="levenshtein" {% if diversity_ranker == 'levenshtein' %}selected{% endif %}>Levenshtein</option>
                            <option value="euclidean" {% if diversity_ranker == 'euclidean' %}selected{% endif %}>Euclidean</option>
                            <option value="diff" {% if diversity_ranker == 'diff' %}selected{% endif %}>Diff</option>
                        </select>
                    </div>
                    <div class="slider-container">
                        <label class="slider-label" for="fluency_threshold">Fluency Threshold:</label>
                        <input type="range" name="fluency_threshold" class="slider" min="0.5" max="1.0" step="0.01" value="{{ fluency_threshold }}" oninput="this.nextElementSibling.value = this.value">
                        <output>{{ fluency_threshold }}</output>
                    </div>
                </div>
                <input type="submit" value="Paraphrase">
            </form>
            {% if paraphrased_essay %}
                <h2>Paraphrased Essay</h2>
                <pre>{{ paraphrased_essay }}</pre>
            {% endif %}
            <div class="cta-buttons">
                <a href="/subscribe">Subscribe Now</a>
                <a href="/contact" class="secondary">Contact Us</a>
            </div>
            <div class="description">
                <h3>What is NexByte's Paraphrasing Tool?</h3>
                <p>The NexByte Paraphrasing Tool is an innovative online tool for converting AI-generated content into human-like writing. This programme, also known as the NexByte AI Text Converter, efficiently rewrites content written by AI writers such as ChatGPT, Google Bard, Microsoft Bing, Claude, QuillBot, Grammarly, Jasper.ai, Copy.ai, and any other AI text generator. It ensures that the text is free of robotic tones, rendering it indistinguishable from human writing.</p>
                <p>Our application employs advanced proprietary algorithms to preserve the original content and context of the text while improving readability and Search Engine Optimisation (SEO) potential. The content created with NexByte Paraphrasing Tool is completely plagiarism-free and undetectable by all existing AI detectors on the market.</p>
                
                <h3>What Does "Paraphrasing AI Text" Mean?</h3>
                <p>Paraphrasing AI text entails transforming AI-generated content into writing that appears more naturally human. This technique entails making the language more interesting, accessible, and clear to human readers while removing any robotic tones.</p>
                <p>NexByte's method for humanising AI text includes:</p>
                <ul>
                    <li><strong>Natural Language Use:</strong> Ensure that the material flows organically and reads easily.</li>
                    <li><strong>Empathy and Understanding:</strong> Adding a human element to make things more relatable.</li>
                    <li><strong>Personalisation:</strong> Tailoring the text to individual audiences and settings.</li>
                    <li><strong>Engagement:</strong> Making the information more intriguing and interactive.</li>
                    <li><strong>Clarity and Simplicity:</strong> Ensure that the text is easy to read and understand.</li>
                    <li><strong>Ethical and Cultural Sensitivity:</strong> Ensure that the content adheres to all cultural and ethical norms.</li>
                </ul>

                <h3>How Can We Paraphrase AI Text Online for Free?</h3>
                <p>Using the NexByte Paraphrasing Tool is simple and intuitive. Follow these easy steps to turn your AI-generated writing into human-like content:</p>
                <ul>
                    <li><strong>Open the NexByte Paraphrasing Tool:</strong> Navigate to NexByte Paraphrasing Tool using your choice web browser. Our programme works with all major browsers.</li>
                    <li><strong>To enter AI-generated text:</strong> Simply paste it into the webpage's input text form.</li>
                    <li><strong>To customise the paraphrasing process:</strong> Adjust preferences such as the diversity ranking method (Levenshtein, Euclidean, or Diff) and fluency threshold.</li>
                    <li><strong>To start the paraphrasing process:</strong> Click the "Paraphrase" button. The tool will begin to convert the AI-generated text into human-like text. Please be patient; this may take some time.</li>
                    <li><strong>After reviewing and editing:</strong> The final output text will be presented. Review the text, and if required, alter the settings before repeating the process until you are satisfied with the outcome.</li>
                    <li><strong>Use the Text:</strong> Copy the relevant text, make any necessary adjustments, and use it in your projects.</li>
                    <li><strong>Click the "Paraphrase Again" button:</strong> To begin a new session with different AI-generated input.</li>
                </ul>
                <p>Voila! You now have content that reads naturally, is free of robotic tones, and is undetectable by AI detection software.</p>

                <h3>Why Should I Use NexByte Paraphrasing Tool?</h3>
                <p>NexByte Paraphrasing Tool stands apart because:</p>
                <ul>
                    <li><strong>Advanced Algorithms:</strong> Uses cutting-edge technologies to assure high-quality paraphrase.</li>
                    <li><strong>User-Friendly Interface:</strong> The design is simple and intuitive, allowing for easy use.</li>
                    <li><strong>Customisable Settings:</strong> Users can fine-tune the paraphrasing process to match unique requirements.</li>
                    <li><strong>Reliable Output:</strong> Generates plagiarism-free, SEO-optimized, and human-like text.</li>
                    <li><strong>Free to Use:</strong> Advanced paraphrasing is available online for free, making it accessible to anyone.</li>
                </ul>
                <p>Experience the future of content creation with the NexByte Paraphrasing Tool, which effortlessly transforms AI-generated prose into human-like masterpieces.</p>
            </div>
            <div class="footer">
                &copy; 2024 NexByte. All rights reserved.
            </div>
        </div>
    </body>
    </html>
    """, paraphrased_essay=paraphrased_essay, diversity_ranker=diversity_ranker, fluency_threshold=fluency_threshold)

if __name__ == '__main__':
    app.run(debug=True)
