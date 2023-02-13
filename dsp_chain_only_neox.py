# Original Luca implemetnation of the DSP chain with NEOX20B

# Imports
import os

from langchain.prompts import PromptTemplate, FewShotPromptTemplate
from langchain.chains import LLMChain

from utils import evaluate, ColBERTv2, format_context, extract_last_line

from llm_wrappers.gooseai import GooseAI

from my_keys import my_gooseai_key


# Set environment variables
os.environ["GOOSEAI_API_KEY"] = my_gooseai_key


# Define the prompt templates
train = [('Who produced the album that included a re-recording of "Lithium"?', ['Butch Vig']),
         ('Who was the director of the 2009 movie featuring Peter Outerbridge as William Easton?', ['Kevin Greutert']),
         ('The heir to the Du Pont family fortune sponsored what wrestling team?', ['Foxcatcher', 'Team Foxcatcher', 'Foxcatcher Team']),
         ('In what year was the star of To Hell and Back born?', ['1925']),
         ('Which award did the first book of Gary Zukav receive?', ['U.S. National Book Award', 'National Book Award']),
         ('What city was the victim of Joseph Druces working in?', ['Boston, Massachusetts', 'Boston']),]

dev = [('Who has a broader scope of profession: E. L. Doctorow or Julia Peterkin?', ['E. L. Doctorow', 'E.L. Doctorow', 'Doctorow']),
       ('What documentary about the Gilgo Beach Killer debuted on A&E?', ['The Killing Season']),
       ('Right Back At It Again contains lyrics co-written by the singer born in what city?', ['Gainesville, Florida', 'Gainesville']),
       ('What year was the party of the winner of the 1971 San Francisco mayoral election founded?', ['1828']),
       ('Which author is English: John Braine or Studs Terkel?', ['John Braine']),
       ('Anthony Dirrell is the brother of which super middleweight title holder?', ['Andre Dirrell']),
       ('In which city is the sports nutrition business established by Oliver Cookson based ?', ['Cheshire', 'Cheshire, UK']),
       ('Find the birth date of the actor who played roles in First Wives Club and Searching for the Elephant.', ['February 13, 1980']),
       ('Kyle Moran was born in the town on what river?', ['Castletown', 'Castletown River']),
       ("What is the name of one branch of Robert D. Braun's speciality?", ['aeronautical engineering', 'astronautical engineering', 'aeronautics', 'astronautics']),
       ("Where was the actress who played the niece in the Priest film born?", ['Surrey', 'Guildford, Surrey']),
       ('Name the movie in which the daughter of Noel Harrison plays Violet Trefusis.', ['Portrait of a Marriage']),
       ('What year was the father of the Princes in the Tower born?', ['1442'])
       ]


# Initialize training and dev sets
train = [{'question': q, 'answer': a[0]} for q, a in train]
dev = [{'question': q, 'answers': a} for q, a in dev]


# Initialize LLM and retrieval model
llm = GooseAI(temperature=0)
rm = ColBERTv2('http://ec2-44-228-128-229.us-west-2.compute.amazonaws.com:8893/api/search')


# QA prompt template
qa_template = """
Question: {question}
Answer: {answer}
"""

example_prompt = PromptTemplate(
    input_variables=['question', 'answer'],
    template=qa_template,
)


# First hop prompt template
search_retrieval_template_first_hop = """
Write a search query that will help answer a complex question.
---
Follow the following format.
Question: $[the question to be answered]
Rationale: Let's think step by step. To answer this question, we first need to find out $[the missing information]
Search Query: $[a simple question for seeking the missing information]
---
Question: {question}
Rationale: Let's think step by step. To answer this question, we first need to find out"""

first_hop_prompt = PromptTemplate(
    input_variables=['question'],
    template=search_retrieval_template_first_hop,
)

first_hop_chain = LLMChain(llm=llm, prompt=first_hop_prompt)


# Follow-up hop prompt template
search_retrieval_template_followup_hop = """
Write a search query that will help answer a complex question.

---

Follow the following format.

Context:
$[sources that may contain relevant content]

Question: $[the question to be answered]
Rationale: Let's think step by step. Based on the context, we have learned the following. $[information from the context that provides useful clues]
Search Query: $[a simple question for seeking the missing information]

---

Context:
{context}

Question: {question}
Rationale: Let's think step by step. Based on the context, we have learned the following.
"""

followup_hop_prompt = PromptTemplate(
    input_variables=['context', 'question'],
    template=search_retrieval_template_followup_hop,
)

followup_hop_chain = LLMChain(llm=llm, prompt=followup_hop_prompt)


# Few-shot prompt template
prefix = """
Answer questions with short factoid answers. Feel free to ignore irrelevant information given in the questions. You have access to the following tools:

Search: Use this tool to search for information on the internet only when it is not provided in Context.
"""

suffix = """
---

Follow the following format.

Context:
$[sources that may contain relevant content]

Question: $[the question to be answered]

Rationale: Let's think step by step. $[a step-by-step deduction that identifies the correct response, which will be provided below]

Answer: $[a short factoid answer, often between 1 and 5 words]

---

Context:
{context}

Question: {question}

Rationale: Let's think step by step."""


rationale_prompt = FewShotPromptTemplate(
    examples=train,
    example_prompt=example_prompt,
    prefix=prefix,
    suffix=suffix,
    input_variables=['context', 'question'],
    example_separator=''
)

answer_chain = LLMChain(llm=llm, prompt=rationale_prompt)

def run_multihop_chain(question):
    context = []

    # Get first hop retrieval question and context
    first_hop_completion = first_hop_chain.run(question=question)
    retrieval_question_first_hop = extract_last_line(first_hop_completion)
    context.extend(rm(retrieval_question_first_hop, k=2))

    # Get second hop retrieval question and context
    second_hop_completion = followup_hop_chain.run(context=format_context(context), question=question)
    retrieval_question_second_hop = extract_last_line(second_hop_completion)

    context.extend(rm(retrieval_question_second_hop, k=2))

    # Get final answer
    final_completion = answer_chain.run(context=format_context(context), question=question)
    answer = extract_last_line(final_completion)

    return answer

evaluate(run_multihop_chain, dev)