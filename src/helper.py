from raptor_helper import *

load_dotenv()
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]


os.environ["PINECONE_API_KEY"] = st.secrets["PINECONE_API_KEY"]


def get_summary(docs):
    """
    Generates a concise summary of a chargesheet based on the provided details.

    Parameters:
    docs (str): The chargesheet details.

    Returns:
    A generator yielding the summary of the chargesheet in chunks.
    """

    system = """
    Given the details of a chargesheet, generate a concise summary following the structure below:

    1. **Introduction**
    - Case Title
    - Case Number
    - Date
    - Jurisdiction

    2. **Accused Information**
    - Names of Accused
    - Roles
    - Background (if relevant)

    3. **Charges**
    - Sections of Law Invoked
    - Description of Charges

    4. **Incident Summary**
    - Date and Time of Incident
    - Location
    - Brief Description of the Incident

    5. **Evidence Summary**
    - Physical Evidence
    - Witnesses
    - Expert Testimony (if any)

    6. **Investigation Details**
    - Investigating Officer
    - Investigation Process
    - Key Findings

    7. **Legal Proceedings**
    - Charges Filed
    - Court Orders (if any)

    8. **Conclusion**
    - Prosecution's Stance
    - Next Steps
    - Additional Remarks

    9. **References (if any)**
    - Supporting Documents

    Please ensure the summary is clear, accurate, and includes all critical information based on the provided chargesheet details."""

    human = "{docs}"
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])

    summary_chain = prompt | llm | StrOutputParser()

    for chunk in summary_chain.stream({"docs": docs}):
        yield chunk


def get_link(docs):
    """
    Generates a link to search for relevant judgments based on the provided legal chargesheet document.

    Args:
        docs (str): The chargesheet document.

    Returns:
        str: The generated link to search for relevant judgments.

    """

    template = """
    I am providing you with a legal chargesheet document. Your task is to identify and return the single most relevant crime mentioned in the document. Please provide only the name of the crime without any additional explanation or context.

    Chargesheet:
    {docs}
    """
    prompt = ChatPromptTemplate.from_template(template)

    link_chain = prompt | llm | StrOutputParser()
    out = link_chain.invoke({"docs": docs})
    link = f"https://indiankanoon.org/search/?formInput={out}+doctypes:judgments"
    return link


def scrape_jina_ai(url: str) -> str:

    response = requests.get("https://r.jina.ai/" + url)
    return response.text


def get_past_judgement_heading(text_list):
    pattern = r"\[(.*?vs.*?on\s+\d{1,2}\s+\w+,\s+\d{4})\]"
    headings = re.findall(pattern, text_list)
    return headings


def past_judgement_link(text_list):
    case_pattern = r"\[Full Document\]\((https://.*?)\)"
    matches = re.findall(case_pattern, text_list)

    docs_link = ["https://r.jina.ai/" + link for link in matches]
    return docs_link


def get_similar_cases_summary(judgement_link):
    """
    This function generates a concise summary of similar legal cases based on the provided questions.

    Parameters:
        questions (list): A list of questions related to the cases.

    Returns:
        A generator yielding the summary of the similar cases in chunks.
    """

    template = """
  

  Given the details of a past legal document, generate a concise summary following the structure below:
  \n\n\nDocument : {context}\n\n\n

1. **Document Overview**
   - Title
   - Date
   - Jurisdiction

2. **Background**
   - Context
   - Parties Involved

3. **Key Issues**
   - Primary Legal Issues
   - Relevant Laws

4. **Summary of Arguments**
   - Arguments by Plaintiff/Prosecution
   - Arguments by Defendant

5. **Evidence Presented**
   - Key Evidence
   - Supporting Documents

6. **Findings and Reasoning**
   - Court’s Findings
   - Legal Reasoning

7. **Decision/Outcome**
   - Ruling
   - Penalties/Sentences

8. **Impact/Significance**
   - Legal Precedent
   - Implications

9. **Conclusion**
   - Summary of the Outcome
   - Next Steps

10. **References**
   - Cited Cases
   - Related Documents

Ensure the summary is clear, accurate, and includes all critical information based on the provided details of the past legal document.
Dont give any links at all
  """
    prompt = ChatPromptTemplate.from_template(template)

    rag_chain = prompt | llm | StrOutputParser()

    response = requests.get(judgement_link)
    docs = response.text
    token_length = count_tokens(docs)
    if token_length > 131000:
        yield (
            "Document too long. Please refer to the original document at the above link "
        )
        return  
    for chunk in rag_chain.stream({"context": docs}):

        yield chunk


def strategy(docs):
    """
    Generates a comprehensive legal strategy for a lawyer based on the provided chargesheet details.

    Parameters:
    docs (str): The chargesheet details.

    Returns:
    A generator yielding the comprehensive legal strategy in chunks.
    """

    system = """
   Based on the details provided in the current chargesheet, generate a comprehensive legal strategy for the lawyer to apply in court. The strategy should include the following elements:

   1. **Case Overview**
   - Brief summary of the charges and key facts.

   2. **Legal Analysis**
   - Identification of the strongest and weakest aspects of the charges.
   - Relevant legal precedents and laws that could support the defense or prosecution.

   3. **Argument Development**
   - Suggested arguments to emphasize during the trial.
   - Counter-arguments to anticipate from the opposing side and how to address them.

   4. **Evidence Strategy**
   - How to present or challenge the key evidence mentioned in the chargesheet.
   - Recommendations for additional evidence or expert testimony that could strengthen the case.

   5. **Witness Management**
   - Key witnesses to focus on, and how to handle their testimonies in court.
   - Strategies for cross-examining opposing witnesses.

   6. **Risk Mitigation**
   - Potential risks or challenges in the case and how to mitigate them.
   - Contingency plans for unexpected developments during the trial.

   7. **Courtroom Tactics**
   - Suggestions for courtroom demeanor, timing of arguments, and interactions with the judge and jury.
   - Recommendations for opening and closing statements.

   8. **Conclusion**
   - Summary of the overall strategy and the desired outcome.

   Please ensure the strategy is practical, aligned with legal principles, and tailored to the specifics of the chargesheet provided.
   """

    human = "{docs}"
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])

    strategy_chain = prompt | llm | StrOutputParser()
    for chunk in strategy_chain.stream({"docs": docs}):

        yield chunk


def extract_text_from_pdf(pdf_file):
    """
    Extracts the text from a given PDF file.

    Parameters:
        pdf_file (file): The PDF file to extract text from.

    Returns:
        str: The extracted text from the PDF file.
    """

    text = ""
    pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        text += page.get_text()
    return text


def raptor_retriever(docs_text: str, index_name: str):
    """
    Retrieves a retriever object for a given document text and index name.

    Parameters:
        docs_text (str): The text content of the document.
        index_name (str): The name of the index to retrieve.

    Returns:
        retriever: A retriever object for the given index.
    """

    pc = chromadb.PersistentClient(path="./chromadb")
    existing_indexes = [c.name for c in pc.list_collections()]
    if index_name not in existing_indexes:
        text_splitter = RecursiveCharacterTextSplitter(
            separators=[
                "\n\n",
                "\n",
                "\n\n\n",
                "\n\n\n\n",
            ],
            chunk_size=1000,
            chunk_overlap=200,
        )
        leaf_texts = text_splitter.split_text(docs_text)
        results = recursive_embed_cluster_summarize(leaf_texts, level=1, n_levels=3)
        all_texts = leaf_texts.copy()

        for level in sorted(results.keys()):
            # Extract summaries from the current level's DataFrame
            summaries = results[level][1]["summaries"].tolist()
            # Extend all_texts with the summaries from the current level
            all_texts.extend(summaries)

        vectorstore = Chroma(
            client=pc,
            collection_name=index_name,
            embedding_function=embd,
            persist_directory="./chromadb",
        )

        vectorstore.add_texts(all_texts)
    else:
        vectorstore = Chroma(
            client=pc,
            collection_name=index_name,
            embedding_function=embd,
            persist_directory="./chromadb",
        )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    return retriever


def raptor(retriever, question: str):
    """
    Generate an assistant response to a given question using a given retriever.

    Args:
        retriever (Retriever): The retriever object used to retrieve relevant context.
        question (str): The question to answer.

    Yields:
        str: A chunk of the assistant's response.

    The assistant is an assistant for question-answering tasks. It uses the provided context to answer the question. If the context does not provide the answer and the assistant is confident about it, it may ignore the context. If the answer is not clear from the context and the assistant is unsure itself, it does not attempt to answer. If the assistant needs to go in-depth to answer the question, it provides a detailed response.

    The assistant generates its response using a template that includes the question and the context. The context is retrieved using the provided retriever and formatted using the format_docs function. The assistant uses a language model (llm) to generate the response. The response is parsed using the StrOutputParser.

    The assistant generates its response in chunks and yields each chunk.

    """

    template = """
   You are an assistant for question-answering tasks. Use the following context to answer the question. If the context does not provide the answer and you are confident about it, you may ignore the context.

    If the answer is not clear from the context and you are unsure yourself, do not attempt to answer. If you need to go in-depth to answer the question, provide a detailed response.
    Do not provide an answer if the context does not provide the answer. 

    Question: {question}

    Context: {context}

    Answer:


    """

    prompt = ChatPromptTemplate.from_template(template)

    rag_chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    for chunk in rag_chain.stream(question):
        yield chunk


def prediction(docs):
    system = """You are an Al trained to act like a judge in the Indian Supreme Court.
        You will be provided with either a chargesheet or an appeal document.
        If the document is a chargesheet, predict the likely outcome regarding the defendant's guilt, responding with either "GUILTY" or "INNOCENT."
        If the document is an appeal, decide whether the appeal should be "ACCEPTED" or "REJECTED" based on the merits of the case,
        considering legal precedents, the strength of the arguments, and the application of relevant laws.
        Please answer only with either "GUILTY," "INNOCENT," "ACCEPTED," or "REJECTED" based on the type of document provided."""
    human = "{docs}"
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    prediction_chain = prompt | llm | StrOutputParser()

    return prediction_chain.invoke({"docs": docs})


def raptor_retriever_pinecone(docs_text: str, index_name: str):
    """docs_text is just a string of the coontents of the pdf"""
    pc = Pinecone()
    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
    if index_name not in existing_indexes:
        text_splitter = RecursiveCharacterTextSplitter(
            separators=[
                "\n\n",
                "\n",
                "\n\n\n",
                "\n\n\n\n",
            ],
            chunk_size=1000,
            chunk_overlap=200,
        )
        leaf_texts = text_splitter.split_text(docs_text)
        results = recursive_embed_cluster_summarize(leaf_texts, level=1, n_levels=3)
        all_texts = leaf_texts.copy()

        for level in sorted(results.keys()):
            # Extract summaries from the current level's DataFrame
            summaries = results[level][1]["summaries"].tolist()
            # Extend all_texts with the summaries from the current level
            all_texts.extend(summaries)

        pc.create_index(
            name=index_name,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        while not pc.describe_index(index_name).status["ready"]:
            time.sleep(1)
        vectorstore = PineconeVectorStore(index_name=index_name, embedding=embd)
        vectorstore.from_texts(texts=all_texts, embedding=embd, index_name=index_name)
    else:
        vectorstore = PineconeVectorStore(index_name=index_name, embedding=embd)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    return retriever


def count_tokens(text, model="gpt-4"):
    # Initialize the tokenizer for the given model
    enc = tiktoken.encoding_for_model(model)

    # Tokenize the text
    tokens = enc.encode(text)

    # Return the number of tokens
    return len(tokens)
