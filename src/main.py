from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    PromptTemplate
)
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import Qdrant
from langchain.output_parsers import PydanticOutputParser
from qdrant_client.http import models as rest
from pydantic import BaseModel, Field
from langchain.document_loaders.csv_loader import CSVLoader


import csv
from typing import Dict, List, Optional
from langchain.document_loaders.base import BaseLoader
from langchain.docstore.document import Document


loader = CSVLoader(
    file_path="./src/dataset_small.csv", source_column="title")

data = loader.load()
