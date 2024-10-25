import whisper
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import PyPDF2
import requests

chat_llm = OllamaLLM(model="llama3.2")
embedding_function = OllamaEmbeddings(model='nomic-embed-text')  
model = whisper.load_model("small")
app = FastAPI()
origins = ["*"]

def retrieve_from_chroma(query, vector_store):
    docs = vector_store.similarity_search(query, k=5) 
    return docs


def generate_answer(query, vector_store, chat_llm):
    relevant_docs = retrieve_from_chroma(query, vector_store)
    context = "\n".join([doc.page_content for doc in relevant_docs])

    prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
    response = chat_llm.invoke(prompt)
    return response


def getQuery(question, currentChapter):
    print("Current Chapter: ",currentChapter)
    vector_store = Chroma(embedding_function=embedding_function, collection_name=currentChapter, persist_directory="./chromadb")
    answer = generate_answer(question, vector_store, chat_llm)
    return answer


def vectorAudioTranscribe(audio, currentChapter):
    r = requests.get(audio, stream=True)

    with open('/tmp/audio.mp3', 'wb') as fd:
        for chunk in r.iter_content(2000):
            fd.write(chunk)
    text = model.transcribe('/tmp/audio.mp3')
    
    # print("Transcription: ", text.text)
    vector_store = Chroma(embedding_function=embedding_function, collection_name=currentChapter, persist_directory="./chromadb")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    # texts = text_splitter.split_text(text.text)
    # docs = [Document(page_content=chunk) for chunk in texts]
    # vector_store.add_documents(docs)
    # print("Audio Transcribed Successfully")
    return "Audio Transcribed Successfully"


def extract_text_from_pdf(pdf_file):
    r = requests.get(pdf_file, stream=True)

    with open('/tmp/metadata.pdf', 'wb') as fd:
        for chunk in r.iter_content(2000):
            fd.write(chunk)
    reader = PyPDF2.PdfReader('/tmp/metadata.pdf')
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    # print(text)
    return text

def PdfUpload(pdf, currentChapter):
    text = extract_text_from_pdf(pdf)
    vector_store = Chroma(embedding_function=embedding_function, collection_name=currentChapter, persist_directory="./chromadb")
    print("Storing in ",currentChapter)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_text(text)
    
    docs = [Document(page_content=chunk) for chunk in texts]
    print(currentChapter)
    vector_store.add_documents(docs)
    vector_store.persist()
    print("PDF Uploaded Successfully")
    return "PDF Uploaded Successfully"

class Message(BaseModel):
    message: str
    currentChapter: str
    currentClass: str


class FileData(BaseModel):
    file: str
    classId: str
    chapterId: str

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root(message):
    print(message) 
    return {"message": "Hello World"}

@app.post("/chat")
async def chat(item: Message):
    print("Current Chapter: ",item.currentChapter)
    return {"message": getQuery(item.message, item.currentChapter)}


@app.post("/audio")
async def audio(item: FileData):
    print("Current Chapter: ",item.file)

    return {"message": vectorAudioTranscribe(item.file, item.chapterId)}

@app.post("/pdf")
async def pdf(item: FileData):
    print("Current Chapter: ",item.chapterId)
    return {"message": PdfUpload(item.file, item.chapterId)}
