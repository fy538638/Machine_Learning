# python3 -m venv ai-exp-audiobook-venv
# pip install openai gradio langchain gtts pytube python-dotenv unstructured tabulate pdf2image
# pip install git+https://github.com/openai/whisper.git

import os
import openai
import gradio as gr
from gtts import gTTS
import whisper
from pytube import YouTube

from langchain.document_loaders import UnstructuredFileLoader
from langchain.chains.summarize import load_summarize_chain
from langchain import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter

from dotenv import load_dotenv

import requests
from bs4 import BeautifulSoup
import openai
import re


load_dotenv()

openai.api_key = os.getenv("OPEN_AI_KEY") #Personal

llm = OpenAI(openai_api_key = os.getenv("OPEN_AI_KEY"))

model = whisper.load_model('base')


def extract_audio_stream_1(youtube_video_url):
    youtube_video = YouTube(youtube_video_url)
    print(youtube_video.title)
    print(youtube_video.thumbnail_url)
    print(youtube_video.streams.filter(only_audio=True).first())

    stream = youtube_video.streams.filter(only_audio=True).first()

    youtube_audio_file = youtube_video_url.split('?')[-1].split('=')[-1] + '.mp4'
    print(f"Audiobook Filename: {youtube_audio_file}")

    if stream:
        stream.download(filename=youtube_audio_file)

    return youtube_audio_file


def translate_2(youtube_audio_file):
    text = model.transcribe(youtube_audio_file, language='Korean', task='translate')
    with open("translation.txt", "w") as file:
        file.write(text['text'])
    return text['text']



def scrape_website_11(url):
    def get_page(url):
        # Send a GET request to the website
        res = requests.get(url, headers={'User-Agent': 'Mozilla'})
        soup = BeautifulSoup(res.text, 'html.parser')
        return soup

    def collect_text(soup):
        text = ''
        para_text = soup.find_all('p')
        print(f"paragraphs text = \n {para_text}")        
        for para in para_text:
            text += f"{para.text}\n\n"
        return text
    
    def clean(text):
        rep = {"<br>": "\n", "<br/>": "\n", "<li>":  "\n"}
        rep = dict((re.escape(k), v) for k, v in rep.items())
        pattern = re.compile("|".join(rep.keys()))
        text = pattern.sub(lambda m: rep[re.escape(m.group(0))], text)
        text = re.sub('\<(.*?)\>', '', text)
        return text

    soup = get_page(url)
    text = collect_text(soup)
    cleaned_text = clean(text)
    
    with open("scraped.txt", "w",encoding="utf-8") as file:
        file.write(cleaned_text)
    
    return cleaned_text

    



# def translate_22(text_file):
#     f = open(text_file, 'r')
#     contents = f.read()
#     from googletrans import Translator
#     file_translate = Translator()
#     result = file_translate.translate(contents, dest='fr')
#     with open("translation.txt", "w") as file:
#         file.write(text['text'])
#     return text['text']



# def summarize_text(transcribed_file):
#     loader = UnstructuredFileLoader(transcribed_file.name)
#     document = loader.load()

#     char_text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0) 
#     docs = char_text_splitter.split_documents(document)
#     print(f"Total chunks to be processed in this long document : {len(docs)}")

#     # Map reduce method
#     chain = load_summarize_chain(llm, chain_type="map_reduce", return_intermediate_steps=True)
#     summary_with_steps = chain({"input_documents": docs}, return_only_outputs=True)

#     summary_text = summary_with_steps['intermediate_steps']
#     summary_cleaned_text = ' '.join(summary_text)

#     print(f"Summary using Map-reduce method with intermediate steps: \n{summary_cleaned_text}\n\n")
#     with open("summary_map_reduce_extended.txt", "w") as summary_extended_file:
#         summary_extended_file.write(summary_cleaned_text)
    
#     with open("summary_map_reduce_concise.txt", "w") as summary_concise_file:
#         summary_concise_file.write(summary_with_steps['output_text'])

#     return summary_cleaned_text



with gr.Blocks() as demo:
    gr.Markdown("<h1>Korean Language Process Project</h1> <h3>This app has the following features backed by AI models.</h3> <h6>1. If you are a KPop lover or would like to check other videos which are in Korean but have no subtitles yet, you can extract audio from Youtube Video URL</h6> <h6>2. Translate Audio to Text</h6> <h6>3. Scrape content from such websites too.</h6> <h6>")
    with gr.Tab("Extract Audio from Youtube video URL"):
        with gr.Row():
            youtube_video_url = gr.Textbox(label="Youtube Video URL")
            youtube_audio_file = gr.File(type="file")
        download_audio_button = gr.Button("Download Youtube Video into Audio")
    with gr.Tab("Translate Audio"):
        with gr.Row():
            audio_file = gr.Audio(type="filepath")    
            text = gr.TextArea(label="Translated Text")
        translate_button = gr.Button("Translate")    
    with gr.Tab("Extract text from websites"):
        with gr.Row():
            news_url = gr.Textbox(label="Website URL") 
            scraped = gr.TextArea(label='Texts')
        scrape_button = gr.Button("Scrape")   
    # with gr.Tab("Summarize Text"):
    #     with gr.Row():
    #         transcribed_file = gr.File(type="file")
    #         summary_text = gr.TextArea(label="Summary")
    #     summarize_button = gr.Button("Summarize")

    

    download_audio_button.click(extract_audio_stream_1, inputs=youtube_video_url, outputs=youtube_audio_file)
    translate_button.click(translate_2, inputs=audio_file, outputs=text)
    scrape_button.click(scrape_website_11, inputs=news_url, outputs=scraped)
    # summarize_button.click(summarize_text, inputs=transcribed_file, outputs=summary_text)

    
demo.launch()
