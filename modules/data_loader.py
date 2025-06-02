import json
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from config import PDF_DIR, CHUNK_STORE, CHUNK_SIZE, CHUNK_OVERLAP, VECTOR_STORE

class PDFLoader:
    def __init__(self, 
                 pdf_dir: str = PDF_DIR,
                 output_file: str = CHUNK_STORE,
                 chunk_size: int = CHUNK_SIZE,
                 chunk_overlap: int = CHUNK_OVERLAP,
                 vector_store_path: str = VECTOR_STORE):
        self.pdf_dir = Path(pdf_dir)
        self.output_file = Path(output_file)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.vector_store_path = vector_store_path
        self.vector_store = None
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

        self.chunker = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ".", ",", " "],
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len
        )

    def load_index(self, path: str):
        """
        Load a FAISS index from disk.
        
        Args:
            path: Directory path containing the saved index
        """
        try:
            self.vector_store = FAISS.load_local(
                path, 
                self.embeddings,
                allow_dangerous_deserialization=True  # Safe since we created the index
            )
            print(f"Loaded vector store from {path}")
        except Exception as e:
            print(f"Error loading vector store: {str(e)}")
            raise

    async def load_pdfs(self):
        pdf_files = list(self.pdf_dir.glob("*.pdf"))
        all_chunks = []

        with open(self.output_file, "w", encoding="utf-8") as out:
            for doc_id, pdf_path in enumerate(pdf_files):
                loader = PyPDFLoader(pdf_path, mode = "page")
                docs = await loader.aload()

                pg = 0
                total_chunks = 0
                skip = False

                for doc in docs:
                    pg += 1
                    text = doc.page_content.strip()

                    if not text or skip:
                        continue

                    lines = text.lower().splitlines()
                    for l in lines:
                        if l.startswith("reference") or l.startswith("references") or l.startswith("bibliography") or l.startswith("acknowledgements"):
                            skip = True
                            break

                    if skip:
                        continue

                    chunks = self.chunker.split_text(text)
                    for i, chunk in enumerate(chunks):
                        data = {
                            "chunk_id": f"d{doc_id:02}p{pg:04}c{i+1:02}",
                            "doc_id": doc_id+1,
                            "doc": pdf_path.name,
                            "page": pg,
                            "text": chunk.strip()
                        }
                        total_chunks += 1
                        json.dump(data, out, ensure_ascii=False)
                        out.write("\n")
                        all_chunks.append(data)
                        
                print(f"\nDocument: {doc_id+1}\nPages: {pg}\nChunks: {total_chunks}\n")
            print(f"{len(pdf_files)} PDF files saved to {self.output_file}")

        # Create vector store
        if all_chunks:
            try:
                # Create documents for FAISS using LangChain's Document class
                documents = []
                for chunk in all_chunks:
                    documents.append(Document(
                        page_content=chunk['text'],
                        metadata={
                            'chunk_id': chunk['chunk_id'],
                            'doc_id': chunk['doc_id'],
                            'doc': chunk['doc'],
                            'page': chunk['page']
                        }
                    ))

                # Create FAISS vector store
                self.vector_store = FAISS.from_documents(
                    documents,
                    self.embeddings
                )

                # Save vector store if path is provided
                if self.vector_store_path:
                    self.vector_store.save_local(self.vector_store_path)
                    print(f"Vector store saved to {self.vector_store_path}")

            except Exception as e:
                print(f"Error creating vector store: {str(e)}")

    def search_similar(self, query: str, k: int = 4):
        """
        Search for similar chunks using the vector store.
        
        Args:
            query: The search query
            k: Number of results to return
            
        Returns:
            List of dictionaries containing similar chunks and their metadata
        """
        if not self.vector_store:
            raise ValueError("Vector store not initialized. Please load PDFs first.")
        
        results = self.vector_store.similarity_search_with_score(query, k=k)
        return [
            {
                'text': doc.page_content,
                'metadata': doc.metadata,
                'score': score
            }
            for doc, score in results
        ]