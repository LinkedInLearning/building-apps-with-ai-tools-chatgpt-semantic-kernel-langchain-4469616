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


import csv
from typing import Dict, List, Optional
from langchain.document_loaders.base import BaseLoader
from langchain.docstore.document import Document

# Advanced Version


class CSVLoader(BaseLoader):
    """Loads a CSV file into a list of documents.

    Each document represents one row of the CSV file. Every row is converted into a
    key/value pair and outputted to a new line in the document's page_content.

    The source for each document loaded from csv is set to the value of the
    `file_path` argument for all doucments by default.
    You can override this by setting the `source_column` argument to the
    name of a column in the CSV file.
    The source of each document will then be set to the value of the column
    with the name specified in `source_column`.

    Output Example:
        .. code-block:: txt

            column1: value1
            column2: value2
            column3: value3
    """

    def __init__(
        self,
        file_path: str,
        source_column: Optional[str] = None,
        metadata_columns: Optional[List[str]] = None,   # < ADDED
        csv_args: Optional[Dict] = None,
        encoding: Optional[str] = None,
    ):
        self.file_path = file_path
        self.source_column = source_column
        self.encoding = encoding
        self.csv_args = csv_args or {}
        self.metadata_columns = metadata_columns        # < ADDED

    def load(self) -> List[Document]:
        """Load data into document objects."""

        docs = []
        with open(self.file_path, newline="", encoding=self.encoding) as csvfile:
            csv_reader = csv.DictReader(
                csvfile, **self.csv_args)  # type: ignore
            for i, row in enumerate(csv_reader):
                content = "\n".join(
                    f"{k.strip()}: {v.strip()}" for k, v in row.items())
                try:
                    source = (
                        row[self.source_column]
                        if self.source_column is not None
                        else self.file_path
                    )
                except KeyError:
                    raise ValueError(
                        f"Source column '{self.source_column}' not found in CSV file."
                    )
                metadata = {"source": source, "row": i}
                # ADDED TO SAVE METADATA
                if self.metadata_columns:
                    for k, v in row.items():
                        if k in self.metadata_columns:
                            metadata[k] = v
                # END OF ADDED CODE
                doc = Document(page_content=content, metadata=metadata)
                docs.append(doc)

        return docs


class BookSearch(BaseModel):
    year: str = Field(description="year book was written")
    genre: str = Field(description="genre of book")


parser = PydanticOutputParser(pydantic_object=BookSearch)

prompt = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={
        "format_instructions": parser.get_format_instructions()},
)

loader = CSVLoader(
    file_path="./src/dataset_small.csv", source_column="title", metadata_columns=["categories", "published_year"])


def get_parsed_result(book_request):
    _input = prompt.format_prompt(query=book_request)

    model = ChatOpenAI()
    output = model.predict(_input.to_string())

    parsed = parser.parse(output)
    return parsed


def create_filter(parsed):
    qdrant_filter = rest.Filter(
        must=[
            rest.FieldCondition(
                key="metadata.categories",
                match=rest.MatchValue(value=parsed.genre),
            ),
            rest.FieldCondition(
                key="metadata.published_year",
                match=rest.MatchValue(value=parsed.year),
            )
        ]
    )
    return qdrant_filter


data = loader.load()

embeddings = OpenAIEmbeddings()

quadrant_docsearch = Qdrant.from_documents(
    data,
    embeddings,
    location=":memory:",
    collection_name="book"
)


while True:
    user_input = input("Hi im an AI librarian what can I help you with?\n")

    parsed_result = get_parsed_result(user_input)
    book_request = "You are a librarian. Help the user answer their question. Do not provide the ISBN." +\
        f"\nUser:{user_input}"

    # advanced version, place inside as_retriever()
    # search_kwargs = {'filter': qdrant_filter}

    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(), chain_type="stuff", retriever=quadrant_docsearch.as_retriever(
            search_kwargs={'filter': create_filter(parsed_result)}), return_source_documents=True)

    result = qa({"query": book_request})
    print(len(result['source_documents']))
    print(result["result"])
