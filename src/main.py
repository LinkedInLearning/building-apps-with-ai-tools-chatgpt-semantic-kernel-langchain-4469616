from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    PromptTemplate
)
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field


class Person(BaseModel):
    first_name: str = Field(description="first name")
    last_name: str = Field(description="last name")
    dob: str = Field(description="date of birth")


class PeopleList(BaseModel):
    people: list[Person] = Field(description="A list of people")
