# %%
import logging
import sys
from llama_index.core import SummaryIndex #type: ignore 
from llama_index.readers.web import SimpleWebPageReader #type: ignore
from IPython.display import Markdown, display

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# NOTE: the html_to_text=True option requires html2text to be installed

PAUL_GRAHAM_WORKED_URL = "http://paulgraham.com/worked.html"
WIKI_NEW_YORK_URL = "https://en.wikipedia.org/wiki/New_York_City"
WEB_PAGE_URL = PAUL_GRAHAM_WORKED_URL

# %%
documents = SimpleWebPageReader(html_to_text=True).load_data(
    [WEB_PAGE_URL]
)
documents[0]
index = SummaryIndex.from_documents(documents, show_progress=True)
# set Logging to DEBUG for more detailed outputs
query_engine = index.as_query_engine()
response = query_engine.query("What did the author do growing up?")
display(Markdown(f"<b>{response}</b>"))
# %%
