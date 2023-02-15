import os
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate, FewShotPromptTemplate
from langchain.chains import LLMChain
from utils import evaluate, ColBERTv2, format_context, extract_last_line

from my_keys import my_open_ai_key, my_api_key, my_cse_id

os.environ["OPENAI_API_KEY"] = my_open_ai_key
os.environ["GOOGLE_CSE_ID"] = my_cse_id
os.environ["GOOGLE_API_KEY"] = my_api_key

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
llm = OpenAI(temperature=0)
rm = ColBERTv2('http://ec2-44-228-128-229.us-west-2.compute.amazonaws.com:8893/api/search')

qa_template = """
Question: {question}
Answer: {answer}
"""

example_prompt = PromptTemplate(
    input_variables=['question', 'answer'],
    template=qa_template,
)

prefix = """
Answer questions with short factoid answers. Feel free to ignore irrelevant information given in the questions.
"""

suffix = """
---

Follow the following format.

Context: 
$[sources that may contain relevant content]

Question: $[the question to be answered]

Answer: $[a short factoid answer, often between 1 and 5 words.]

---

Context:
{context}

Question: {question}

Thought: Let's think step by step."""

rationale_prompt = FewShotPromptTemplate(
    examples=train,
    example_prompt=example_prompt,
    prefix=prefix,
    suffix=suffix,
    input_variables=['context', 'question'],
    example_separator=''
)

answer_chain = LLMChain(llm=llm, prompt=rationale_prompt)

def run_ralm_chain(question):
    context = []
    context.extend(rm(question['question'], k=3))
    print(format_context(context))
    # Get final answer
    final_completion = answer_chain.run(context=format_context(context), question=question)
    print(final_completion)
    answer = extract_last_line(final_completion)
    return answer

evaluate(run_ralm_chain, dev)