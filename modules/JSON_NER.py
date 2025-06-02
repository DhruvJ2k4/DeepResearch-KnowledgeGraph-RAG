from langchain_community.chat_models import ChatOpenAI
import os

import json
from pathlib import Path
import time
#from langchain_ollama import OllamaLLM
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from config import CHUNK_STORE, BLOCK_SIZE, MODEL, PROMPT, ENTITY_PATH

from langchain_core.output_parsers import PydanticOutputParser
from pydantic import RootModel
from typing import Dict, List
from config import (
    NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD,
    TOGETHER_API_KEY, TOGETHER_API_BASE, LLM_MODEL, LLM_TEMPERATURE
)

class EntityResponse(RootModel[Dict[str, List[str]]]):
    pass

class Chunks_NER:
    def __init__(self, 
                 prompt: str = PROMPT,
                 chunk_store: str = CHUNK_STORE,
                 model: str = MODEL):
        self.model = model
        self.prompt = prompt

        

        # Add your API key here or load from env
        os.environ["TOGETHER_API_KEY"] = TOGETHER_API_KEY
        self.llm = ChatOpenAI(
            temperature=LLM_TEMPERATURE,
            openai_api_key=TOGETHER_API_KEY,
            openai_api_base=TOGETHER_API_BASE,
            model_name=LLM_MODEL,
        )


        #self.llm = OllamaLLM(model=self.model)
        # self.prompt_template = PromptTemplate(template=prompt,
        #                                input_variables=['known_entities', 'batch_texts'])
        self.parser = PydanticOutputParser(pydantic_object=EntityResponse)
        format_instructions = self.parser.get_format_instructions()

        self.prompt_template = PromptTemplate(
            template=PROMPT + "\n{format_instructions}",
            input_variables=['known_entities', 'batch_texts'],
            partial_variables={"format_instructions": format_instructions}
        )

        self.chain = self.prompt_template |self.llm | self.parser

    def load_chunks(self, json_path: str = CHUNK_STORE):        
        json_path = Path(json_path)
        out = dict()

        with open(json_path, "r", encoding="utf-8") as f:
            chunks = list()
            for line in f:
                data = json.loads(line)
                chunks.append([data["chunk_id"], data["text"]])
        return chunks



    def Extract_Entities(self):
        entities = set()
        chunks = self.load_chunks()
        dump = dict()
        with open(ENTITY_PATH, "w", encoding="utf-8") as f:
            for i in range(0, len(chunks), BLOCK_SIZE):
                start = time.time()
                batch = chunks[i:i + BLOCK_SIZE]
                
                prompt_input = {
                    "known_entities": list(entities)[-100:],
                    "batch_texts": "\n".join([f"{cid}: {text}" for cid, text in batch])
                }

                try:
                    response = self.chain.invoke(prompt_input)
                except Exception as e:
                    print(f"Error processing batch {i // BLOCK_SIZE + 1}: {e}")
                    continue
                
                for chunk_id, entity in response.root.items():
                    normalized = [e.strip().lower() for e in entity]
                    dump[chunk_id] = normalized
                    entities.update(normalized)
                
                end = time.time()
                print(f"Batch {i // BLOCK_SIZE + 1} took {end - start:.2f} seconds")

            json.dump(dump, f, indent=2, ensure_ascii=False) # change later 