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


model = ChatOpenAI(model="gpt-4")
people_data = model.predict(
    "Generate a list of 10 fake peoples information. Only return the list. Each person should have a first name, last name and date of birth.")

parser = PydanticOutputParser(pydantic_object=PeopleList)

prompt = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={
        "format_instructions": parser.get_format_instructions()},
)

_input = prompt.format_prompt(query=people_data)

model = ChatOpenAI()
output = model.predict(_input.to_string())

parsed = parser.parse(output)
print(parsed)
