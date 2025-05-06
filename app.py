import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
import re
import string
import docx
import PyPDF2
import os
import spacy
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from textblob import TextBlob
import gensim
from gensim import corpora
from wordcloud import WordCloud

class CooperationMiner:
    def __init__(self, root):
        self.root = root
        self.root.title("CooperationMiner - Text Mining for Social Dilemma Analysis")
        self.root.geometry("1200x800")
        self.root.configure(bg="#f0f0f0")
        
        self.setup_nltk()
        self.setup_spacy()
        self.create_widgets()
        
        self.text_data = ""
        self.processed_text = ""
        self.file_path = ""
        self.tokens = []
        self.sentences = []
        self.cooperation_keywords = [
            "collaborate", "cooperate", "together", "mutual", "share", "help",
            "support", "assist", "benefit", "joint", "collective", "common",
            "agreement", "consensus", "compromise", "trust", "reciprocate",
            "fairness", "equitable", "partnership", "alliance", "team"
        ]
        self.non_cooperation_keywords = [
            "defect", "cheat", "selfish", "individual", "alone", "betray",
            "abandon", "exploit", "deceive", "trick", "mislead", "unfair",
            "greedy", "distrust", "suspicious", "compete", "rivalry", "opponent"
        ]
        
    def setup_nltk(self):
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet')
        try:
            nltk.data.find('sentiment/vader_lexicon')
        except LookupError:
            nltk.download('vader_lexicon')
            
        # Remove the problematic punkt_tab check completely
    
    def setup_spacy(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            import subprocess
            subprocess.call([
                "python", "-m", "spacy", "download", "en_core_web_sm"
            ])
            self.nlp = spacy.load("en_core_web_sm")
    
    def create_widgets(self):
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.create_menu()
        self.create_file_frame()
        self.create_notebook()
        self.create_status_bar()
    
    def create_menu(self):
        menu_bar = tk.Menu(self.root)
        
        file_menu = tk.Menu(menu_bar, tearoff=0)
        file_menu.add_command(label="Open File", command=self.open_file)
        file_menu.add_command(label="Save Results", command=self.save_results)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        menu_bar.add_cascade(label="File", menu=file_menu)
        
        analysis_menu = tk.Menu(menu_bar, tearoff=0)
        analysis_menu.add_command(label="Basic Text Stats", command=self.analyze_basic_stats)
        analysis_menu.add_command(label="Sentiment Analysis", command=self.analyze_sentiment)
        analysis_menu.add_command(label="Cooperation Analysis", command=self.analyze_cooperation)
        analysis_menu.add_command(label="Topic Modeling", command=self.analyze_topics)
        analysis_menu.add_command(label="Entity Recognition", command=self.analyze_entities)
        menu_bar.add_cascade(label="Analysis", menu=analysis_menu)
        
        help_menu = tk.Menu(menu_bar, tearoff=0)
        help_menu.add_command(label="About", command=self.show_about)
        menu_bar.add_cascade(label="Help", menu=help_menu)
        
        self.root.config(menu=menu_bar)
    
    def create_file_frame(self):
        file_frame = ttk.LabelFrame(self.main_frame, text="File Selection")
        file_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(file_frame, text="Browse...", command=self.open_file).pack(side=tk.LEFT, padx=5, pady=5)
        self.file_label = ttk.Label(file_frame, text="No file selected")
        self.file_label.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.X, expand=True)
        ttk.Button(file_frame, text="Analyze", command=self.run_full_analysis).pack(side=tk.RIGHT, padx=5, pady=5)
    
    def create_notebook(self):
        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.text_tab = ttk.Frame(self.notebook)
        self.basic_stats_tab = ttk.Frame(self.notebook)
        self.sentiment_tab = ttk.Frame(self.notebook)
        self.cooperation_tab = ttk.Frame(self.notebook)
        self.topic_tab = ttk.Frame(self.notebook)
        self.visualization_tab = ttk.Frame(self.notebook)
        
        self.notebook.add(self.text_tab, text="Text View")
        self.notebook.add(self.basic_stats_tab, text="Basic Stats")
        self.notebook.add(self.sentiment_tab, text="Sentiment")
        self.notebook.add(self.cooperation_tab, text="Cooperation Analysis")
        self.notebook.add(self.topic_tab, text="Topic Modeling")
        self.notebook.add(self.visualization_tab, text="Visualizations")
        
        self.setup_text_tab()
        self.setup_basic_stats_tab()
        self.setup_sentiment_tab()
        self.setup_cooperation_tab()
        self.setup_topic_tab()
        self.setup_visualization_tab()
    
    def setup_text_tab(self):
        self.text_display = scrolledtext.ScrolledText(self.text_tab, wrap=tk.WORD, width=80, height=30)
        self.text_display.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    def setup_basic_stats_tab(self):
        self.stats_display = scrolledtext.ScrolledText(self.basic_stats_tab, wrap=tk.WORD, width=80, height=30)
        self.stats_display.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    def setup_sentiment_tab(self):
        self.sentiment_frame = ttk.Frame(self.sentiment_tab)
        self.sentiment_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.sentiment_display = scrolledtext.ScrolledText(self.sentiment_frame, wrap=tk.WORD, width=40, height=15)
        self.sentiment_display.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.sentiment_fig_frame = ttk.Frame(self.sentiment_frame)
        self.sentiment_fig_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    def setup_cooperation_tab(self):
        self.cooperation_frame = ttk.Frame(self.cooperation_tab)
        self.cooperation_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.cooperation_display = scrolledtext.ScrolledText(self.cooperation_frame, wrap=tk.WORD, width=40, height=15)
        self.cooperation_display.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.cooperation_fig_frame = ttk.Frame(self.cooperation_frame)
        self.cooperation_fig_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    def setup_topic_tab(self):
        self.topic_frame = ttk.Frame(self.topic_tab)
        self.topic_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.topic_display = scrolledtext.ScrolledText(self.topic_frame, wrap=tk.WORD, width=40, height=15)
        self.topic_display.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.topic_fig_frame = ttk.Frame(self.topic_frame)
        self.topic_fig_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    def setup_visualization_tab(self):
        self.viz_frame = ttk.Frame(self.visualization_tab)
        self.viz_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        viz_controls = ttk.Frame(self.viz_frame)
        viz_controls.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(viz_controls, text="Visualization Type:").pack(side=tk.LEFT, padx=5, pady=5)
        self.viz_type = tk.StringVar(value="wordcloud")
        viz_options = ttk.Combobox(viz_controls, textvariable=self.viz_type)
        viz_options['values'] = ('wordcloud', 'cooperation_over_time', 'emotion_distribution', 'topic_distribution')
        viz_options.pack(side=tk.LEFT, padx=5, pady=5)
        
        ttk.Button(viz_controls, text="Generate", command=self.generate_visualization).pack(side=tk.LEFT, padx=5, pady=5)
        
        self.viz_fig_frame = ttk.Frame(self.viz_frame)
        self.viz_fig_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    def create_status_bar(self):
        self.status_bar = ttk.Label(self.root, text="Ready", relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def open_file(self):
        file_types = [
            ("All Supported Files", "*.txt *.pdf *.docx *.csv"),
            ("Text Files", "*.txt"),
            ("PDF Files", "*.pdf"),
            ("Word Documents", "*.docx"),
            ("CSV Files", "*.csv"),
            ("All Files", "*.*")
        ]
        
        self.file_path = filedialog.askopenfilename(filetypes=file_types)
        
        if self.file_path:
            self.file_label.config(text=os.path.basename(self.file_path))
            self.status_bar.config(text=f"Loading file: {os.path.basename(self.file_path)}")
            self.root.update_idletasks()
            
            self.text_data = self.read_file(self.file_path)
            self.text_display.delete(1.0, tk.END)
            self.text_display.insert(tk.END, self.text_data)
            
            self.status_bar.config(text=f"Loaded file: {os.path.basename(self.file_path)}")
    
    def read_file(self, file_path):
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == '.txt':
            with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
                return file.read()
        
        elif ext == '.pdf':
            text = ""
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num in range(len(pdf_reader.pages)):
                    text += pdf_reader.pages[page_num].extract_text() + "\n"
            return text
        
        elif ext == '.docx':
            doc = docx.Document(file_path)
            return "\n".join([paragraph.text for paragraph in doc.paragraphs])
        
        elif ext == '.csv':
            df = pd.read_csv(file_path)
            text_columns = df.select_dtypes(include=['object']).columns
            
            if len(text_columns) > 0:
                return "\n".join(df[text_columns[0]].astype(str).tolist())
            else:
                return df.to_string()
        
        else:
            return f"Unsupported file format: {ext}"
    
    def preprocess_text(self, text):
        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()
        
        # Use the nltk.tokenize.sent_tokenize function directly
        # which will use the default punkt tokenizer
        self.sentences = sent_tokenize(text)
        
        tokens = []
        for sentence in self.sentences:
            words = word_tokenize(sentence.lower())
            words = [word for word in words if word.isalnum()]
            words = [word for word in words if word not in stop_words]
            words = [lemmatizer.lemmatize(word) for word in words]
            tokens.extend(words)
        
        self.tokens = tokens
        self.processed_text = " ".join(tokens)
        return self.processed_text
    
    def run_full_analysis(self):
        if not self.text_data:
            self.status_bar.config(text="No text data to analyze.")
            return
        
        self.status_bar.config(text="Analyzing text...")
        self.root.update_idletasks()
        
        self.preprocess_text(self.text_data)
        
        self.analyze_basic_stats()
        self.analyze_sentiment()
        self.analyze_cooperation()
        self.analyze_topics()
        self.analyze_entities()
        self.generate_visualization()
        
        self.status_bar.config(text="Analysis complete!")
    
    def analyze_basic_stats(self):
        if not self.text_data:
            return
        
        stats = {}
        stats["Total Characters"] = len(self.text_data)
        stats["Total Words"] = len(word_tokenize(self.text_data))
        stats["Total Sentences"] = len(sent_tokenize(self.text_data))
        stats["Average Word Length"] = sum(len(word) for word in word_tokenize(self.text_data)) / max(1, len(word_tokenize(self.text_data)))
        stats["Average Sentence Length (words)"] = stats["Total Words"] / max(1, stats["Total Sentences"])
        stats["Unique Words"] = len(set(word.lower() for word in word_tokenize(self.text_data) if word.isalnum()))
        
        word_freq = Counter(word.lower() for word in word_tokenize(self.text_data) if word.isalnum())
        most_common = word_freq.most_common(10)
        
        self.stats_display.delete(1.0, tk.END)
        self.stats_display.insert(tk.END, "BASIC TEXT STATISTICS\n")
        self.stats_display.insert(tk.END, "====================\n\n")
        
        for stat, value in stats.items():
            self.stats_display.insert(tk.END, f"{stat}: {value:.2f}\n" if isinstance(value, float) else f"{stat}: {value}\n")
        
        self.stats_display.insert(tk.END, "\nMost Common Words:\n")
        for word, count in most_common:
            self.stats_display.insert(tk.END, f"{word}: {count}\n")
    
    def analyze_sentiment(self):
        if not self.sentences:
            return
        
        sia = SentimentIntensityAnalyzer()
        
        sentiment_scores = []
        for sentence in self.sentences:
            score = sia.polarity_scores(sentence)
            sentiment_scores.append({
                'sentence': sentence,
                'compound': score['compound'],
                'positive': score['pos'],
                'negative': score['neg'],
                'neutral': score['neu']
            })
        
        df_sentiment = pd.DataFrame(sentiment_scores)
        overall_sentiment = df_sentiment['compound'].mean()
        
        self.sentiment_display.delete(1.0, tk.END)
        self.sentiment_display.insert(tk.END, "SENTIMENT ANALYSIS\n")
        self.sentiment_display.insert(tk.END, "=================\n\n")
        
        sentiment_label = "Positive" if overall_sentiment > 0.05 else "Negative" if overall_sentiment < -0.05 else "Neutral"
        self.sentiment_display.insert(tk.END, f"Overall Sentiment: {sentiment_label} ({overall_sentiment:.4f})\n\n")
        
        self.sentiment_display.insert(tk.END, "Average Sentiment Metrics:\n")
        self.sentiment_display.insert(tk.END, f"Positive: {df_sentiment['positive'].mean():.4f}\n")
        self.sentiment_display.insert(tk.END, f"Negative: {df_sentiment['negative'].mean():.4f}\n")
        self.sentiment_display.insert(tk.END, f"Neutral: {df_sentiment['neutral'].mean():.4f}\n\n")
        
        self.sentiment_display.insert(tk.END, "Most Positive Sentences:\n")
        top_positive = df_sentiment.nlargest(3, 'compound')
        for _, row in top_positive.iterrows():
            self.sentiment_display.insert(tk.END, f"- {row['sentence']} (score: {row['compound']:.4f})\n")
        
        self.sentiment_display.insert(tk.END, "\nMost Negative Sentences:\n")
        top_negative = df_sentiment.nsmallest(3, 'compound')
        for _, row in top_negative.iterrows():
            self.sentiment_display.insert(tk.END, f"- {row['sentence']} (score: {row['compound']:.4f})\n")
        
        for widget in self.sentiment_fig_frame.winfo_children():
            widget.destroy()
        
        fig, ax = plt.subplots(figsize=(6, 4))
        sentiment_data = [df_sentiment['positive'].mean(), df_sentiment['negative'].mean(), df_sentiment['neutral'].mean()]
        ax.bar(['Positive', 'Negative', 'Neutral'], sentiment_data, color=['green', 'red', 'blue'])
        ax.set_ylabel('Score')
        ax.set_title('Average Sentiment Scores')
        
        canvas = FigureCanvasTkAgg(fig, master=self.sentiment_fig_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def analyze_cooperation(self):
        if not self.sentences:
            return
        
        cooperation_scores = []
        for sentence in self.sentences:
            words = word_tokenize(sentence.lower())
            coop_words = sum(1 for word in words if word in self.cooperation_keywords)
            non_coop_words = sum(1 for word in words if word in self.non_cooperation_keywords)
            
            if len(words) > 0:
                coop_ratio = (coop_words - non_coop_words) / len(words)
            else:
                coop_ratio = 0
            
            cooperation_scores.append({
                'sentence': sentence,
                'cooperation_score': coop_ratio,
                'coop_words': coop_words,
                'non_coop_words': non_coop_words
            })
        
        df_coop = pd.DataFrame(cooperation_scores)
        overall_coop = df_coop['cooperation_score'].mean()
        
        self.cooperation_display.delete(1.0, tk.END)
        self.cooperation_display.insert(tk.END, "COOPERATION ANALYSIS\n")
        self.cooperation_display.insert(tk.END, "====================\n\n")
        
        coop_label = "Cooperative" if overall_coop > 0 else "Non-cooperative" if overall_coop < 0 else "Neutral"
        self.cooperation_display.insert(tk.END, f"Overall Cooperation Tendency: {coop_label} ({overall_coop:.4f})\n\n")
        
        total_coop_words = df_coop['coop_words'].sum()
        total_non_coop_words = df_coop['non_coop_words'].sum()
        
        self.cooperation_display.insert(tk.END, f"Total Cooperative Keywords: {total_coop_words}\n")
        self.cooperation_display.insert(tk.END, f"Total Non-cooperative Keywords: {total_non_coop_words}\n\n")
        
        self.cooperation_display.insert(tk.END, "Most Cooperative Sentences:\n")
        top_coop = df_coop.nlargest(3, 'cooperation_score')
        for _, row in top_coop.iterrows():
            self.cooperation_display.insert(tk.END, f"- {row['sentence']} (score: {row['cooperation_score']:.4f})\n")
        
        self.cooperation_display.insert(tk.END, "\nLeast Cooperative Sentences:\n")
        bottom_coop = df_coop.nsmallest(3, 'cooperation_score')
        for _, row in bottom_coop.iterrows():
            self.cooperation_display.insert(tk.END, f"- {row['sentence']} (score: {row['cooperation_score']:.4f})\n")
        
        for widget in self.cooperation_fig_frame.winfo_children():
            widget.destroy()
        
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(['Cooperative', 'Non-cooperative'], [total_coop_words, total_non_coop_words], color=['green', 'red'])
        ax.set_ylabel('Count')
        ax.set_title('Cooperative vs Non-cooperative Language')
        
        canvas = FigureCanvasTkAgg(fig, master=self.cooperation_fig_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def analyze_topics(self):
        if not self.processed_text:
            return
        
        vectorizer = TfidfVectorizer(max_features=1000)
        tfidf_matrix = vectorizer.fit_transform([self.processed_text])
        
        feature_names = vectorizer.get_feature_names_out()
        
        lda = LatentDirichletAllocation(n_components=5, random_state=42)
        lda.fit(tfidf_matrix)
        
        topics = []
        for topic_idx, topic in enumerate(lda.components_):
            top_words_idx = topic.argsort()[:-11:-1]  
            top_words = [feature_names[i] for i in top_words_idx]
            topics.append({
                'topic_id': topic_idx,
                'top_words': top_words,
                'weight': topic.sum()
            })
        
        self.topic_display.delete(1.0, tk.END)
        self.topic_display.insert(tk.END, "TOPIC MODELING\n")
        self.topic_display.insert(tk.END, "=============\n\n")
        
        for topic in topics:
            self.topic_display.insert(tk.END, f"Topic {topic['topic_id']+1} (Weight: {topic['weight']:.2f}):\n")
            self.topic_display.insert(tk.END, ", ".join(topic['top_words']) + "\n\n")
        
        for widget in self.topic_fig_frame.winfo_children():
            widget.destroy()
        
        fig, ax = plt.subplots(figsize=(6, 4))
        topic_weights = [topic['weight'] for topic in topics]
        ax.bar(range(1, len(topics)+1), topic_weights)
        ax.set_xlabel('Topic')
        ax.set_ylabel('Weight')
        ax.set_title('Topic Weights')
        ax.set_xticks(range(1, len(topics)+1))
        
        canvas = FigureCanvasTkAgg(fig, master=self.topic_fig_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def analyze_entities(self):
        if not self.text_data:
            return
        
        doc = self.nlp(self.text_data)
        
        entities = []
        for ent in doc.ents:
            entities.append({
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char
            })
        
        entity_counts = Counter([ent['label'] for ent in entities])
        
        self.status_bar.config(text=f"Found {len(entities)} named entities.")
    
    def generate_visualization(self):
        viz_type = self.viz_type.get()
        
        for widget in self.viz_fig_frame.winfo_children():
            widget.destroy()
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        if viz_type == 'wordcloud':
            if not self.processed_text:
                return
            
            wordcloud = WordCloud(width=800, height=600, 
                                  background_color='white',
                                  max_words=100).generate(self.processed_text)
            
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            ax.set_title('Word Cloud')
        
        elif viz_type == 'cooperation_over_time':
            if not self.sentences:
                return
            
            scores = []
            for sentence in self.sentences:
                words = word_tokenize(sentence.lower())
                coop_words = sum(1 for word in words if word in self.cooperation_keywords)
                non_coop_words = sum(1 for word in words if word in self.non_cooperation_keywords)
                
                if len(words) > 0:
                    coop_ratio = (coop_words - non_coop_words) / len(words)
                else:
                    coop_ratio = 0
                
                scores.append(coop_ratio)
            
            window_size = max(5, len(scores) // 20)
            smoothed_scores = pd.Series(scores).rolling(window=window_size, center=True).mean().fillna(0).tolist()
            
            ax.plot(smoothed_scores)
            ax.set_xlabel('Sentence Position')
            ax.set_ylabel('Cooperation Score')
            ax.set_title('Cooperation Tendency Over Time')
            ax.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        
        elif viz_type == 'emotion_distribution':
            if not self.sentences:
                return
            
            emotions = {'joy': 0, 'sadness': 0, 'anger': 0, 'fear': 0, 'surprise': 0, 'disgust': 0}
            
            for sentence in self.sentences:
                blob = TextBlob(sentence)
                polarity = blob.sentiment.polarity
                subjectivity = blob.sentiment.subjectivity
                
                if polarity > 0.5:
                    emotions['joy'] += 1
                elif polarity < -0.5:
                    emotions['sadness'] += 1
                elif polarity < -0.2:
                    emotions['anger'] += 1
                elif subjectivity > 0.8:
                    emotions['surprise'] += 1
                elif polarity < 0 and subjectivity > 0.5:
                    emotions['disgust'] += 1
                elif polarity < -0.1:
                    emotions['fear'] += 1
            
            emotions_filtered = {k: v for k, v in emotions.items() if v > 0}
            
            if emotions_filtered:
                ax.pie(emotions_filtered.values(), labels=emotions_filtered.keys(), autopct='%1.1f%%')
                ax.set_title('Emotion Distribution')
            else:
                ax.text(0.5, 0.5, "Not enough emotion data to visualize", 
                        horizontalalignment='center', verticalalignment='center')
        
        elif viz_type == 'topic_distribution':
            if not self.processed_text:
                return
            
            vectorizer = TfidfVectorizer(max_features=1000)
            tfidf_matrix = vectorizer.fit_transform([self.processed_text])
            
            lda = LatentDirichletAllocation(n_components=5, random_state=42)
            lda.fit(tfidf_matrix)
            
            topic_weights = lda.components_.sum(axis=1)
            topic_weights = topic_weights / topic_weights.sum()
            
            ax.pie(topic_weights, labels=[f'Topic {i+1}' for i in range(len(topic_weights))], autopct='%1.1f%%')
            ax.set_title('Topic Distribution')
        
        canvas = FigureCanvasTkAgg(fig, master=self.viz_fig_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def save_results(self):
        if not self.file_path:
            self.status_bar.config(text="No file selected")
            return
        
        file_types = [
            ("Text Files", "*.txt"),
            ("CSV Files", "*.csv"),
            ("All Files", "*.*")
        ]
        
        save_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=file_types)
        
        if not save_path:
            return
        
        with open(save_path, 'w', encoding='utf-8') as file:
            file.write("COOPERATION MINER ANALYSIS RESULTS\n")
            file.write("================================\n\n")
            file.write(f"File analyzed: {os.path.basename(self.file_path)}\n")
            file.write(f"Date of analysis: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            file.write("BASIC STATISTICS\n")
            file.write("---------------\n")
            file.write(f"Total Characters: {len(self.text_data)}\n")
            file.write(f"Total Words: {len(word_tokenize(self.text_data))}\n")
            file.write(f"Total Sentences: {len(sent_tokenize(self.text_data))}\n")
            file.write(f"Unique Words: {len(set(word.lower() for word in word_tokenize(self.text_data) if word.isalnum()))}\n\n")
            
            file.write("SENTIMENT ANALYSIS\n")
            file.write("-----------------\n")
            sia = SentimentIntensityAnalyzer()
            overall_score = sia.polarity_scores(self.text_data)
            file.write(f"Positive: {overall_score['pos']:.4f}\n")
            file.write(f"Negative: {overall_score['neg']:.4f}\n")
            file.write(f"Neutral: {overall_score['neu']:.4f}\n")
            file.write(f"Compound: {overall_score['compound']:.4f}\n\n")
            
            file.write("COOPERATION ANALYSIS\n")
            file.write("--------------------\n")
            coop_words = sum(1 for word in self.tokens if word in self.cooperation_keywords)
            non_coop_words = sum(1 for word in self.tokens if word in self.non_cooperation_keywords)
            file.write(f"Cooperative Keywords: {coop_words}\n")
            file.write(f"Non-cooperative Keywords: {non_coop_words}\n")
            if len(self.tokens) > 0:
                coop_ratio = (coop_words - non_coop_words) / len(self.tokens)
                file.write(f"Cooperation Ratio: {coop_ratio:.4f}\n\n")
            
            file.write("\n© 2025 Pejman Ebrahimi - All rights reserved")
        
        self.status_bar.config(text=f"Results saved to {os.path.basename(save_path)}")
    
    def show_about(self):
        about_window = tk.Toplevel(self.root)
        about_window.title("About CooperationMiner")
        about_window.geometry("400x300")
        about_window.resizable(False, False)
        
        ttk.Label(about_window, text="CooperationMiner", font=("Arial", 16, "bold")).pack(pady=10)
        ttk.Label(about_window, text="Text Mining for Social Dilemma Analysis").pack()
        ttk.Label(about_window, text="Version 1.0").pack(pady=5)
        
        desc_frame = ttk.Frame(about_window)
        desc_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        description = ("CooperationMiner is a text mining application designed to analyze "
                      "communication patterns in social dilemma situations. It identifies "
                      "cooperative language, sentiment, and topics to provide insights "
                      "into how communication facilitates cooperation in social dilemmas.")
        
        ttk.Label(desc_frame, text=description, wraplength=350, justify=tk.CENTER).pack(pady=10)
        
        ttk.Label(about_window, text="© 2025 Pejman Ebrahimi - All rights reserved").pack(pady=10)
        
        ttk.Button(about_window, text="Close", command=about_window.destroy).pack(pady=10)


def main():
    import nltk
    import os
    
    # Create a local nltk_data directory if it doesn't exist
    if not os.path.exists('nltk_data'):
        os.makedirs('nltk_data')
        
    # Download all required NLTK data to the local directory
    nltk.download('punkt', download_dir='nltk_data')
    nltk.download('stopwords', download_dir='nltk_data')
    nltk.download('wordnet', download_dir='nltk_data')
    nltk.download('vader_lexicon', download_dir='nltk_data')
    
    # Add the local path to NLTK's search paths
    nltk.data.path.insert(0, './nltk_data')
    
    root = tk.Tk()
    app = CooperationMiner(root)
    root.eval('tk::PlaceWindow . center')
    
    footer = ttk.Label(root, text="© 2025 Pejman Ebrahimi - All rights reserved", 
                      background="#f0f0f0", foreground="#666666")
    footer.pack(side=tk.BOTTOM, fill=tk.X, pady=2)
    
    root.mainloop()

if __name__ == "__main__":
    main()