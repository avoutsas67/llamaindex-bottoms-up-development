# %% [markdown]
# # Faithfulness Evaluator
# ## Introduction
# ### [Faithfulness Evaluator Example](https://docs.llamaindex.ai/en/stable/examples/evaluation/faithfulness_eval)

# %% [markdown]
# ## 1. Import Libraries
# %%
import os
import sys
import nest_asyncio #type: ignore
from dotenv import load_dotenv, find_dotenv  # type: ignore

# ## Using the OpenAI LLM with the VectorStoreIndex
from openai import __version__ as openai_version  # type: ignore
from llama_index.core import __version__ as llama_index_version  # type: ignore

from llama_index.core import (
    VectorStoreIndex,
    Settings,
    SimpleDirectoryReader,
    Response,
) # type: ignore
from llama_index.llms.openai import OpenAI #type: ignore
from llama_index.embeddings.openai import OpenAIEmbedding #type: ignore
from llama_index.core.schema import NodeWithScore #type: ignore
from llama_index.core.evaluation import FaithfulnessEvaluator #type: ignore
from llama_index.core.node_parser import SentenceSplitter #type: ignore
from llama_index.core.evaluation import EvaluationResult #type: ignore
from llama_index.readers.web import SimpleWebPageReader #type: ignore
from IPython.display import display #type: ignore
import pandas as pd 

# ## 2. Initialisation

# attach to the same event-loop
nest_asyncio.apply()

# Load environment variables
_ = load_dotenv(find_dotenv())  # read local .env file

print(f"Python version: {sys.version}")
print(f"OpenAI version: {openai_version}")
print(f"llamaindex version: {llama_index_version}")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# %%
pd.set_option("display.max_colwidth", 0)

# gpt-4
Settings.llm = OpenAI(temperature=0, model="gpt-4")
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

evaluator_gpt4 = FaithfulnessEvaluator()

# %% [markdown]
# ## 3. Load Data and Create Vector Index
# documents = SimpleDirectoryReader("./test_wiki_data/").load_data()
PAUL_GRAHAM_WORKED_URL = "http://paulgraham.com/worked.html"
WIKI_NEW_YORK_URL = "https://en.wikipedia.org/wiki/New_York_City"
WEB_PAGE_URL = WIKI_NEW_YORK_URL
documents = SimpleWebPageReader(html_to_text=True).load_data(
    [WEB_PAGE_URL]
)

# create vector index
splitter = SentenceSplitter(chunk_size=512)
vector_index = VectorStoreIndex.from_documents(
    documents, transformations=[splitter], show_progress=True
)

# %% [markdown]
# ## 4. Evaluate Faithfulness
# %%
query_engine = vector_index.as_query_engine()
response_vector = query_engine.query("How did New York City get its name?")
eval_result = evaluator_gpt4.evaluate_response(response=response_vector)

# %% [markdown]
# ## 5. Display Evaluation Results
# %%
def display_eval_df(response: Response, eval_result: EvaluationResult) -> None:
    source_nodes: list[NodeWithScore] = response.source_nodes
    if response.source_nodes == []:
        print("no response!")
        return
    else:
        source_node: NodeWithScore = source_nodes[0]
    eval_df = pd.DataFrame(
        {
            "Response": str(response),
            "Source": source_node.text[:1000] + "...",
            "Evaluation Result": "Pass" if eval_result.passing else "Fail",
            "Reasoning": eval_result.feedback,
        },
        index=[0],
    )
    styled_eval_df = eval_df.style.set_properties(
        **{
            "inline-size": "600px",
            "overflow-wrap": "break-word",
        },
        subset=["Response", "Source"]
    )
    display(styled_eval_df)

display_eval_df(response_vector, eval_result)
# %%
