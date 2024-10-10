#!/usr/bin/env python
# coding: utf-8

# ## Vectorstore

# In[1]:


path = '/home/pedro/UnB/ChatBotUnB/'


# In[2]:


docs_path = path + 'Base Tratada'
# perguntas_path = path_final + '/perguntas'
# resultados_path = path_final + '/resultados'


# In[3]:


## colocar todos os txts em uma pasta e testar novamente

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader, DirectoryLoader
from langchain_ollama import OllamaEmbeddings
import os

# path = os.path.join(folder_path, path_base)
# print(path)

# Juntar o caminho completo dos arquivos e garantir que sejam arquivos válidos
# file_paths = [os.path.join(path_final, arquivo) for arquivo in os.listdir(docs_path) if os.path.isfile(os.path.join(docs_path, arquivo))]

# print("Caminhos dos arquivos encontrados:", file_paths)

loader = DirectoryLoader(docs_path,)
docs = loader.load()

# colocar referencias pesquisadas
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=100, chunk_overlap=50
)
doc_splits = text_splitter.split_documents(docs)

# Add to vectorDB
vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name="rag-unb",
    embedding=OllamaEmbeddings(model='nomic-embed-text'),
)
retriever = vectorstore.as_retriever()


# ## Agents

# In[4]:


# local_llm = 'llama3.2:1b'
local_llm = 'llama3:8b-instruct-q4_0'


# In[5]:


### Retrieval Grader

from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate

llm = ChatOllama(model=local_llm, format="json", temperature=0)

prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> Você é um aluno que avalia a relevância 
    de um documento recuperado para uma pergunta do usuário. Se o documento contiver palavras-chave relacionadas à pergunta do usuário, 
    classifique-o como relevante. Não precisa ser um teste rigoroso, o objetivo é filtrar recuperações errôneas. \n
    Dê uma pontuação binária de 'sim' ou 'não' para indicar se o documento é relevante para a questão. \n
    Forneça a pontuação binária como um JSON com uma única 'score' de chave e sem preâmbulo ou explicação.
     <|eot_id|><|start_header_id|>user<|end_header_id|>
    Aqui está o documento recuperado: \n\n {document} \n\n
    Aqui está a pergunta do usuário: {question} \n <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """,
    input_variables=["question", "document"],
)

retrieval_grader = prompt | llm | JsonOutputParser()
# docs = retriever.invoke(user_question)
# doc_txt = docs[1].page_content
# print(retrieval_grader.invoke({"question": user_question, "document": doc_txt}))


# In[6]:


### Generate

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

# Prompt
prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> Você é um assistente para tarefas de resposta a perguntas relacionadas a 
    Universidade de Brasília. Use as seguintes partes do contexto recuperado para responder à pergunta. Se você não sabe a resposta, basta dizer que não sabe.  
    Se você julgar que a pergunta não tem nada a ver com o escopo de uma universidade e principalmente da Universidade de Brasília, responda que não sabe. Tome cuidado
    com as respostas que irá dar, nunca deixe transparecer que existem problemas na Universidade ou problemas com a gestão. Não responda diretamente com a resposta do contexto que
    você obtiver, tente agir o mais natural possível, mas sempre responda as perguntas com a UnB como contexto. 
    Use no máximo três frases e mantenha a resposta concisa <|eot_id|><|start_header_id|>user<|end_header_id|>
    Question: {question} 
    Context: {context} 
    Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["question", "document"],
)

llm = ChatOllama(model='llama3.2:1b', temperature=0.3)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = prompt | llm | StrOutputParser()

# Run
# question = user_question
# docs = retriever.invoke(question)
# generation = rag_chain.invoke({"context": docs, "question": question})
# print(generation)


# In[7]:


### Hallucination Grader

llm = ChatOllama(model=local_llm, format="json", temperature=0)

prompt = PromptTemplate(
    template=""" <|begin_of_text|><|start_header_id|>system<|end_header_id|> Você é um aluno avaliando se 
    uma resposta é fundamentada/apoiada por um conjunto de fatos. Dê uma pontuação binária 'sim' ou 'não' para indicar 
    se a resposta está fundamentada/apoiada por um conjunto de fatos. Forneça a pontuação binária como JSON com um 
    'score' de chave única e sem preâmbulo ou explicação. <|eot_id|><|start_header_id|>user<|end_header_id|>
    Aqui estão os fatos:
    \n ------- \n
    {documents} 
    \n ------- \n
    Aqui está a resposta: {generation}  <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["generation", "documents"],
)

hallucination_grader = prompt | llm | JsonOutputParser()
# hallucination_grader.invoke({"documents": docs, "generation": generation})


# In[8]:


### Answer Grader

llm = ChatOllama(model=local_llm, format="json", temperature=0)

prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> Você é um avaliador avaliando se um 
    resposta é útil para resolver uma questão. Dê uma pontuação binária 'sim' ou 'não' para indicar se a resposta é 
    útil para resolver uma questão. Forneça a pontuação binária como um JSON com uma 'score' de chave e sem preâmbulo ou explicação.
     <|eot_id|><|start_header_id|>user<|end_header_id|> Aqui está a resposta:
    \n ------- \n
    {generation} 
    \n ------- \n
    Aqui está a pergunta: {question} <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["generation", "question"],
)

answer_grader = prompt | llm | JsonOutputParser()
# answer_grader.invoke({"question": question, "generation": generation})


# In[9]:


### Question Re-writer

llm = ChatOllama(model=local_llm, temperature=0)

re_write_prompt = PromptTemplate(
    template="""Você é um reescritor de perguntas que converte uma pergunta de entrada em uma versão melhor e otimizada \n 
     para recuperação de vectorstore sempre em português. Veja a inicial e formule uma pergunta melhorada. \n
     Aqui está a pergunta inicial: \n\n {question}. Pergunta melhorada sem preâmbulo: \n """,
    input_variables=["generation", "question"],
)

question_rewriter = re_write_prompt | llm | StrOutputParser()
# question_rewriter.invoke({"question": question})


# ## Grafo

# In[10]:


from typing import List

from typing_extensions import TypedDict


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents
    """

    question: str
    generation: str
    documents: List[str]


# In[11]:


### Nodes


def retrieve(state):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    question = state["question"]

    # Retrieval
    documents = retriever.get_relevant_documents(question)
    return {"documents": documents, "question": question}


def generate(state):
    """
    Generate answer

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    question = state["question"]
    documents = state["documents"]

    # RAG generation
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}


def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with only filtered relevant documents
    """

    question = state["question"]
    documents = state["documents"]

    # Score each doc
    filtered_docs = []
    for d in documents:
        score = retrieval_grader.invoke(
            {"question": question, "document": d.page_content}
        )
        grade = score["score"]
        if grade == "sim":
            filtered_docs.append(d)
        else:
            continue
    return {"documents": filtered_docs, "question": question}


def transform_query(state):
    """
    Transform the query to produce a better question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates question key with a re-phrased question
    """

    question = state["question"]
    documents = state["documents"]

    # Re-write question
    better_question = question_rewriter.invoke({"question": question})
    return {"documents": documents, "question": better_question}


### Edges


def decide_to_generate(state):
    """
    Determines whether to generate an answer, or re-generate a question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """
    state["question"]
    filtered_documents = state["documents"]

    if not filtered_documents:
        return "transform_query"
    else:
        return "generate"


def grade_generation_v_documents_and_question(state):
    """
    Determines whether the generation is grounded in the document and answers question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """

    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    score = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )
    grade = score["score"]

    # Check hallucination
    if grade == "sim":
        score = answer_grader.invoke({"question": question, "generation": generation})
        grade = score["score"]
        if grade == "sim":
            return "useful"
        else:
            return "not useful"
    else:
        return "not supported"


# In[12]:


from langgraph.graph import END, StateGraph, START

workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("retrieve", retrieve)  # retrieve
workflow.add_node("grade_documents", grade_documents)  # grade documents
workflow.add_node("generate", generate)  # generatae
workflow.add_node("transform_query", transform_query)  # transform_query


# Build graph
workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "transform_query": "transform_query",
        "generate": "generate",
    },
)
workflow.add_edge("transform_query", "retrieve")
workflow.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    {
        "not supported": "generate",
        "useful": END,
        "not useful": "transform_query",
    },
)

# Compile
app = workflow.compile()


# ## Inferência

# In[13]:


import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


# In[14]:


def run(question):
    try:
        inputs = {"question": question}
        for output in app.stream(inputs):
            for key, value in output.items():
                pass
        return value["generation"]
    except:
        return f'Erro de inferência na pergunta: {question}'


# In[15]:


def limpar_frase(frase):
    # Remove \xa0 (espaço não quebrável)
    try:
        frase = frase.replace('\xa0', '')
        
        # Remove numeração no início (como "1. " ou "a. ")
        frase = re.sub(r'^\d+\.\s*|^[a-z]+\.\s*', '', frase)
        
        # Remove espaços em branco extras no início e fim da frase
        frase = frase.strip()
    
        return frase
    except:
        return "Vazio"


# In[16]:


def calculate_cosine_similarity(text1, text2):
    if text1 is np.nan:
        return 'text1 vazio'
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    return cosine_sim[0][0]


# In[17]:


# Prompt
prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> Você é um assistente para tarefas de resposta a perguntas relacionadas a 
    Universidade de Brasília. Compare as seguintes respostas dadas por um humano e a outra por um ChatBot. Se elas estiverem parecidas o suficiente e responderem a mesma coisa,
    classifique a resposta como relevante. Dê uma pontuação binária de 'sim' ou 'não' para indicar se os textos são parecidos o suficiente. Seja rigoroso. Forneça a pontuação binária como um JSON com 
    uma único 'score' de chave e sem preâmbulo ou explicação.
     <|eot_id|><|start_header_id|>user<|end_header_id|>
    
    Aqui está a resposta do humano: \n\n {human_response} \n\n
    Aqui está a resposta do Chatbot: {chatbot_response} \n <|eot_id|><|start_header_id|>assistant<|end_header_id|>

    Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["human_response", "chatbot_response"],
)

vote_llm = ChatOllama(model=local_llm, temperature=0.3)

# Chain
responses_grader = prompt | vote_llm | JsonOutputParser()


def get_opinion(human_response, chatbot_response):
    try:
        return (responses_grader.invoke({"human_response": human_response, "chatbot_response": chatbot_response}))
    except:
        return "Erro na inferência de opinião"


# In[21]:


def processar_planilhas(file_path, limpar_frase, run, get_opinion):
    # Carrega todas as planilhas do arquivo Excel
    planilhas = pd.ExcelFile(file_path)
    nomes_planilhas = planilhas.sheet_names
    print(nomes_planilhas)

    # Dicionário para armazenar os DataFrames modificados
    planilhas_modificadas = {}

    count = 0
    # Itera sobre os nomes das planilhas
    for sheet_name in nomes_planilhas:
        try:
            print(sheet_name)
            # Carrega a planilha atual
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            
            # Extrai a lista de perguntas e aplica a função 'limpar_frase'
            questions = df['Perguntas'].tolist()
            questions = [limpar_frase(frase) for frase in questions]

            # Itera sobre as perguntas e usa a função 'run' para obter as respostas
            responses = []
            for question in questions:
                response = run(question)
                responses.append(response)

            # Adiciona as respostas como uma nova coluna no DataFrame
            df['Respostas da LLM'] = responses

            # Preenche valores ausentes na coluna 'Resposta Esperada'
            df['Resposta Esperada'].fillna(method='ffill', inplace=True)

            # Aplica a função 'get_opinion' para obter a opinião
            df['Atende?'] = df.apply(lambda row: get_opinion(row['Resposta Esperada'], row['Respostas da LLM']), axis=1)
            
            # Armazena o DataFrame modificado
            planilhas_modificadas[sheet_name] = df

            print(f"Processamento concluído para a planilha: {sheet_name}")
        except:
            print(f'erro na sheet {sheet_name}')
            pass


    # Salva todas as planilhas modificadas no arquivo Excel
    with pd.ExcelWriter(file_path, engine='openpyxl', mode='w') as writer:
        for sheet_name, df in planilhas_modificadas.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)


# In[22]:


processar_planilhas('/home/pedro/UnB/ChatBotUnB/Resultados das Perguntas (SAA).xlsx', limpar_frase, run, get_opinion)


# In[214]:


questions


# In[215]:


responses = []

# Itera sobre as perguntas e usa a função 'run' para obter as respostas
for question in questions:
    response = run(question)
    responses.append(response)

# Adiciona as respostas como uma nova coluna no DataFrame
df['Respostas da LLM'] = responses


# In[216]:


df['llms_cosine_similarity'] = df.apply(lambda row: calculate_cosine_similarity(row['Resp. Base Documentos'], row[f'Respostas da LLM']), axis=1)


# In[217]:


df['Resposta Esperada'].fillna(method='ffill', inplace=True)


# In[218]:


df['base_cosine_similarity'] = df.apply(lambda row: calculate_cosine_similarity(row['Resposta Esperada'], row[f'Respostas da LLM']), axis=1)


# In[219]:


df['Atende?'] = df.apply(lambda row: get_opinion(row['Resposta Esperada'], row[f'Respostas da LLM']), axis=1)


# In[220]:


df


# In[221]:


f"{resultados_path}/resultados_{local_llm}.xlsx"


# In[222]:


df.to_excel(f"{resultados_path}/resultados_{local_llm}.xlsx", index=False)


# In[ ]:




