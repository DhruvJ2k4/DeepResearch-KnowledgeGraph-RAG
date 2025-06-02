# run_ingest.py
import asyncio
from modules.data_loader import PDFLoader
from modules.JSON_NER import Chunks_NER
from modules.KnowledgeGraph import KnowledgeGraph
from modules.agent import SearchAgent

# loader = PDFLoader()
# asyncio.run(loader.load_pdfs())

# Uncomment the following lines to run the NER and Knowledge Graph creation


# ner = Chunks_NER()
# ner.Extract_Entities()  # Extract entities from the loaded chunks

# kg = KnowledgeGraph()
# kg.create_graph()
# kg.close()

# Initialize the agent
agent = SearchAgent()

# Perform a search
response = agent.search("How can I increase my training throughput on my text-guided attention model?", k=5)

# Print the response
print(response)

# Don't forget to close connections when done
agent.close()

