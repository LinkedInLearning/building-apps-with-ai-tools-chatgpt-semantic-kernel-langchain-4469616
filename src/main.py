from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    PromptTemplate
)
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
