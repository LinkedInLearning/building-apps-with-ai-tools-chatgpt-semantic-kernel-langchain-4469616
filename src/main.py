from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain
from langchain.schema import BaseOutputParser


def generate_book_recommendations(book_requests):
    """
    Generate book recommendations based on user requests
    """
    # create templates
    system_template_book_agent = """You are book recommendation agent. Provide a short recommendation based on the user request."""
    system_message_prompt = SystemMessagePromptTemplate.from_template(
        system_template_book_agent)

    human_template_book_agent = "{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(
        human_template_book_agent)

    # create full prompt
    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt])

    chain = LLMChain(
        llm=ChatOpenAI(temperature=1),
        prompt=chat_prompt
    )

    recommendations = []
    for book_request in book_requests:
        recommendations.append(chain.run(book_request))

    return recommendations


def generate_book_requests(n=5) -> list[str]:
    """ Generate book requests
    n: number of requests
    """
    # create templates
    system_template_book_agent = """Generate one utterance for how someone may ask for a {text}. Include a genre and a year. """
    system_message_prompt = SystemMessagePromptTemplate.from_template(
        system_template_book_agent)

    # create full prompt
    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt])

    chain = LLMChain(
        llm=ChatOpenAI(model='gpt-4'),
        prompt=chat_prompt
    )

    results = []

    for _ in range(0, n):
        results.append(chain.run("book"))

    return results


# generate some requests
book_requests = generate_book_requests()
print("The book requests are:\n")
print("\n".join(book_requests))
print("\n")
# get the recommendations
print("The book recommendations are:\n")
recommendations = generate_book_recommendations(book_requests)
print("\n\n".join(recommendations))
print("\n")
