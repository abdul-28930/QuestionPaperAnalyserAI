import openai
import pandas as pd
import PyPDF2
import random
from nltk.tokenize import sent_tokenize
from fpdf import FPDF
import nltk
nltk.download('punkt_tab')

class AIQuestionGenerator:
    def __init__(self, filepath, api_key):
        self.filepath = filepath
        self.api_key = api_key
        openai.api_key = self.api_key
        self.text = self._read_file()
        self.unwanted_phrases = [] 

    def add_unwanted_phrase(self, phrase):
        """Method to add a phrase to the unwanted_phrases array."""
        self.unwanted_phrases.append(phrase.lower())

    def _read_file(self):
        if self.filepath.endswith('.csv'):
            return self._read_csv()
        elif self.filepath.endswith('.pdf'):
            return self._read_pdf()
        else:
            raise ValueError("Unsupported file format. Use CSV or PDF files.")
    
    def _read_csv(self):
        df = pd.read_csv(self.filepath)
        text = ' '.join(df.astype(str).values.flatten())
        return text

    def _read_pdf(self):
        text = ""
        with open(self.filepath, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text()
        return text

    def analyze_text(self):
        sentences = sent_tokenize(self.text)
        return sentences

    def generate_questions(self, num_questions, difficulty):
        sentences = self.analyze_text()
        selected_sentences = random.sample(sentences, min(num_questions, len(sentences)))

        questions = []
        for sentence in selected_sentences:
            prompt = self._generate_prompt(sentence, difficulty)
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",  
                messages=[
                    {"role": "system", "content": "You are a question generator."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=150,
                temperature=0.7
            )
            question = response["choices"][0]["message"]["content"].strip()
            if not self.is_too_specific(question) and not self.contains_unwanted_phrase(question):
                questions.append(question)

        return questions

    def _generate_prompt(self, sentence, difficulty):
        base_prompt = ("Generate general questions based on the following text that focus on broad concepts or practical "
                       "applications, avoiding specific references to textbooks, authors, or publication details. The questions "
                       "should be suitable for an exam setting and should test the understanding of key concepts or practical skills "
                       "without delving into specific details or lists. : '{}'").format(sentence)
        
        if difficulty == 'easy':
            return f"Generate a simple question from the following sentence: {base_prompt}"
        elif difficulty == 'medium':
            return f"Generate a moderate difficulty question from the following sentence: {base_prompt}"
        elif difficulty == 'hard':
            return f"Generate a challenging question from the following sentence: {base_prompt}"
        else:
            raise ValueError("Invalid difficulty level. Choose 'easy', 'medium', or 'hard'.")

    def is_too_specific(self, question):
        specific_terms = ['course objectives', 'syllabus', 'technical terms']
        return any(term in question.lower() for term in specific_terms)

    def contains_unwanted_phrase(self, question):
        return any(phrase in question.lower() for phrase in self.unwanted_phrases)

    def save_questions(self, questions, output_format='pdf'):
        filename = input("Enter the filename (without extension): ")

        if output_format == 'pdf':
            self._save_as_pdf(questions, filename)
        elif output_format == 'excel':
            self._save_as_excel(questions, filename)
        else:
            raise ValueError("Unsupported output format. Choose 'pdf' or 'excel'.")

    def _save_as_pdf(self, questions, filename):
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_font('Arial', size=12)
        for question in questions:
            pdf.multi_cell(0, 10, question)
            pdf.ln()
        pdf.output(f"{filename}.pdf")  

    def _save_as_excel(self, questions, filename):
        df = pd.DataFrame(questions, columns=["Questions"])
        df.to_excel(f"{filename}.xlsx", index=False)  


if __name__ == "__main__":
    filepath = input("Enter file path: ")
    api_key = input("Enter your OpenAI API key: ") 
    num_questions = int(input("Enter the number of questions: "))
    difficulty = input("Enter the difficulty level (easy, medium, hard): ")

    generator = AIQuestionGenerator(filepath, api_key)
   
    generator.add_unwanted_phrase("key topics covered")
    generator.add_unwanted_phrase("common pitfalls")
    generator.add_unwanted_phrase("key considerations")

    questions = generator.generate_questions(num_questions, difficulty)
    output_format = input("Enter the output format (pdf, excel): ")
    generator.save_questions(questions, output_format)
