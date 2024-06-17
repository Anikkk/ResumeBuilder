import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings


if __name__ == '__main__':
    # print('Ingestion..')
    load_dotenv()
    loader = TextLoader("resume_org.txt")
    document = loader.load()
    print("spltting")

    text_splitters = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitters.split_documents(document)
    print(f'Created {len(texts)}')

    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))
    print('Ingesting')

    PineconeVectorStore.from_documents(texts,embeddings,index_name=os.environ.get("INDEX_NAME"))
    print('Finish')