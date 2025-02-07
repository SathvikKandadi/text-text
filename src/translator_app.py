import tkinter as tk
from tkinter import ttk
import speech_recognition as sr
import pyttsx3
import openai
from deep_translator import GoogleTranslator
import os
from dotenv import load_dotenv
load_dotenv()

class TranslationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Speech Translation App")
        self.root.configure(bg='#FFE4E1')  # Light pink background
        
        # Initialize components
        self.setup_gui()
        self.setup_speech_components()
        
        # Load OpenAI API key
        openai.api_key = os.getenv('OPENAI_API_KEY')
        
    def setup_gui(self):
        # Title
        title = tk.Label(self.root, text="Speech to Speech Translation App",
                        font=('Courier', 24), bg='#FFE4E1')
        title.pack(pady=20)
        
        # Language selection
        self.language_var = tk.StringVar(value="German")
        languages = ["German", "Spanish"]
        language_menu = ttk.Combobox(self.root, textvariable=self.language_var,
                                   values=languages)
        language_menu.pack(pady=10)
        
        # Speak button
        speak_btn = tk.Button(self.root, text="Speak",
                            command=self.handle_speech_input,
                            font=('Courier', 12))
        speak_btn.pack(pady=10)
        
        # Display area
        self.display_area = tk.Text(self.root, height=10, width=50,
                                  font=('Courier', 12))
        self.display_area.pack(pady=20)
        
        # Replay button
        replay_btn = tk.Button(self.root, text="Replay Audio",
                             command=self.replay_translation,
                             font=('Courier', 12))
        replay_btn.pack(pady=10)
        
        # Credits
        credits = tk.Label(self.root, text="© Your Name",
                         font=('Courier', 10), bg='#FFE4E1')
        credits.pack(side=tk.BOTTOM, pady=10)

    def setup_speech_components(self):
        self.recognizer = sr.Recognizer()
        self.engine = pyttsx3.init()
        
    def handle_speech_input(self):
        with sr.Microphone() as source:
            self.display_area.delete(1.0, tk.END)
            self.display_area.insert(tk.END, "Listening...\n")
            self.root.update()
            
            try:
                audio = self.recognizer.listen(source, timeout=5)
                text = self.recognizer.recognize_google(audio)
                
                # Get translation from OpenAI
                target_lang = self.language_var.get()
                translated = self.get_openai_translation(text, target_lang)
                
                # Display results
                self.display_area.delete(1.0, tk.END)
                self.display_area.insert(tk.END, f"You said: {text}\n")
                self.display_area.insert(tk.END, 
                    f"Translated text: {translated}\n")
                
                # Speak the translation
                self.speak_translation(translated)
                
            except Exception as e:
                self.display_area.delete(1.0, tk.END)
                self.display_area.insert(tk.END, f"Error: {str(e)}")

    def get_openai_translation(self, text, target_lang):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": f"You are a translator. Translate the following text to {target_lang}."},
                    {"role": "user", "content": text}
                ]
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"OpenAI error: {str(e)}")
            # Fallback to Google Translate
            try:
                translator = GoogleTranslator(source='en', target=self.get_language_code(target_lang))
                return translator.translate(text)
            except Exception as e:
                print(f"Translation error: {str(e)}")
                return "Translation failed"

    def get_language_code(self, language):
        codes = {
            "German": "de",
            "Spanish": "es"
        }
        return codes.get(language, "en")

    def speak_translation(self, text):
        self.engine.say(text)
        self.engine.runAndWait()

    def replay_translation(self):
        last_translation = self.display_area.get(1.0, tk.END).split('\n')[1]
        if last_translation.startswith("Translated text: "):
            text = last_translation.replace("Translated text: ", "")
            self.speak_translation(text)

def main():
    root = tk.Tk()
    app = TranslationApp(root)
    root.mainloop()

if __name__ == "__main__":
    main() 