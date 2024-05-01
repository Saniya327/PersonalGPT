import nltk
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import os
import shutil
import ssl


def load_documents(DATA_PATH):
  loader = DirectoryLoader(DATA_PATH, glob="**/*.txt")
  documents=loader.load()
  return documents


def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    return chunks


def save_to_chroma(chunks: list[Document], CHROMA_PATH):
    #TODO: replace with your API key
    os.environ['OPENAI_API_KEY'] = 'YOUR_API_KEY_HERE'
    # Clear out the database first.
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # Create a new DB from the documents.
    db = Chroma.from_documents(
        chunks, OpenAIEmbeddings(), persist_directory=CHROMA_PATH
    )
    db.persist()
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")


def initialize_db():
  try:
    _create_unverified_https_context = ssl._create_unverified_context
  except AttributeError:
    pass
  else:
    ssl._create_default_https_context = _create_unverified_https_context
  nltk.download('punkt')
  CHROMA_PATH = "chroma"
#TODO: your API key here
  os.environ['OPENAI_API_KEY'] = 'YOUR_API_KEY'
  embedding_function = OpenAIEmbeddings()
  db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
  return db


def get_query_answer(query_text, db):
  PROMPT_TEMPLATE = """
  Answer the question based only on the following context:

  {context}

  ---

  If needed, mention the summary of the context but do not refer to the context assuming I know what the context is. 
  Answer the question based on the above context: {question}
  """

  results = db.similarity_search_with_relevance_scores(query_text, k=3)
  if len(results) == 0 or results[0][1] < 0.7:
    return "Unable to find matching results."
  
  context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
  prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
  prompt = prompt_template.format(context=context_text, question=query_text)

  model = ChatOpenAI()
  response_text = model.predict(prompt)

  sources = [doc.metadata.get("source", None) for doc, _score in results]
  formatted_response = f"Response: {response_text}\nSources: {sources}"
  return formatted_response
