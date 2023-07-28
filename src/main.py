import openai
import numpy as np
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.schema import BaseOutputParser
from langchain.evaluation import QAEvalChain

question_answers = [
    {'question': "When was tea discovered?",
        'answer': "3rd century"},
    {'question': "I'd like a 1 line ice cream slogan",
        'answer': "It's the coolest thing around!"}
]
llm = ChatOpenAI(model="gpt-4")
predictions = []
responses = []
for pairs in question_answers:
    q = pairs["question"]
    response = llm.predict(
        f"Generate the response to the question: {q}. Only print the answer.")
    responses.append(response)
    predictions.append({"result": {response}})

print("\nGenerating text matchs:")

for i in range(0, len(responses)):
    print(question_answers[i]["answer"] == response[i])


resp = openai.Embedding.create(
    input=[r["answer"] for r in question_answers] + responses,
    engine="text-embedding-ada-002")

print("\nGenerating Similarity Score:!")
for i in range(0, len(question_answers)*2, 2):
    embedding_a = resp['data'][i]['embedding']
    embedding_b = resp['data'][len(question_answers)]['embedding']
    similarity_score = np.dot(embedding_a, embedding_b)
    print(similarity_score, similarity_score > 0.8)


print("\nGenerating Self eval:")

# Start your eval chain
eval_chain = QAEvalChain.from_llm(llm)

# Have it grade itself. The code below helps the eval_chain know where the different parts are
graded_outputs = eval_chain.evaluate(question_answers,
                                     predictions,
                                     question_key="question",
                                     prediction_key="result",
                                     answer_key='answer')
print(graded_outputs)
