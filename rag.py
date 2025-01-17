from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.llms import LlamaCpp
from langchain.chains import LLMChain
from langchain.chains.prompt_selector import ConditionalPromptSelector
from langchain.prompts import PromptTemplate

import sqlite3
print(sqlite3.sqlite_version)

loader = PyMuPDFLoader("./Alex.pdf")
PDF_data = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=5)
all_splits = text_splitter.split_documents(PDF_data)
model_name = "sentence-transformers/all-MiniLM-L6-v2"
model_kwargs = {'device': 'cpu'}
embedding = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)
persist_directory = 'db'
vectordb = Chroma.from_documents(documents=all_splits, embedding=embedding, persist_directory=persist_directory)

llm = LlamaCpp(
    model_path="llama-2_q4.gguf",
    n_gpu_layers=100,
    n_batch=512,
    n_ctx=2048,
    f16_kv=True,
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    verbose=True,
)
LLAMA_SEARCH_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""<<SYS>> Assist to provide search result. <</SYS>>
    [INST] Provide an answer.{question} [/INST]""",
)

SEARCH_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""Assist to provide search result. \
        Provide an answer.\
        {question}""",
)

QUESTION_PROMPT = ConditionalPromptSelector(
    default_prompt=SEARCH_PROMPT,
    conditionals=[(lambda llm: isinstance(llm, LlamaCpp), LLAMA_SEARCH_PROMPT)],
)

prompt = QUESTION_PROMPT.get_prompt(llm)
llm_chain = LLMChain(prompt=prompt, llm=llm)
question = "Whats best food in the US?"
llm_chain.invoke({"question": question})

