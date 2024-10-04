from raptor_helper import *




warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def get_summary(docs, llm):
    """
    this gives the summary of the legal document.
    """

    system = """


**Summarize the provided chargesheet details concisely, following this structure:**

1. **Introduction**
   - Case Title: [Insert title]
   - Case Number: [Insert number]
   - Date: [Insert date]
   - Jurisdiction: [Insert jurisdiction]

2. **Accused**
   - Names: [List accused]
   - Roles: [Describe roles]
   - Background: [Provide relevant background]

3. **Charges**
   - Sections: [List sections]
   - Description: [Brief description]

4. **Incident**
   - Date/Time: [Insert date/time]
   - Location: [Insert location]
   - Incident Summary: [Brief summary]

5. **Evidence**
   - Physical: [List evidence]
   - Witnesses: [List witnesses]
   - Expert Testimony: [If applicable]

6. **Investigation**
   - Officer: [Name]
   - Process: [Brief summary]
   - Key Findings: [Summary]

7. **Legal Proceedings**
   - Charges Filed: [List charges]
   - Court Orders: [If any]

8. **Conclusion**
   - Prosecution: [Prosecution's stance]
   - Next Steps: [Outline steps]
   - Remarks: [Additional comments]

9. **References**
   - Documents: [List documents]



    """

    human = "{docs}"
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])

    summary_chain = prompt | llm | StrOutputParser()
    
    for chunk in summary_chain.stream({"docs": docs}):
        yield chunk


def get_link(docs, llm):
    """
    find the one major crime of the document and return the link to the relevant case of indiankanoon.org

    """
    template = """
    You are a legal document classification assistant. Analyze the given legal document and identify which crime(s) from the following list it most likely pertains to:

    - Terrorism
    - Conspiracy Against the State
    - Treason
    - Espionage
    - Sedition
    - Robbery
    - Theft
    - Burglary
    - Arson
    - Murder
    - Manslaughter
    - Homicide
    - Assault
    - Battery
    - Kidnapping
    - Human Trafficking
    - Domestic Violence
    - Child Abuse
    - Extortion
    - Fraud
    - Embezzlement
    - Bribery
    - Forgery
    - Counterfeiting
    - Perjury
    - Obstruction of Justice
    - Money Laundering
    - Tax Evasion
    - Drug Trafficking
    - Possession of Controlled Substances
    - DUI (Driving Under the Influence)
    - Cybercrime
    - Hacking
    - Identity Theft
    - Vandalism
    - Stalking
    - Harassment
    - Prostitution
    - Illegal Possession of Firearms
    - Animal Cruelty
    - Public Intoxication
    - Disorderly Conduct
    - Trespassing
    - Obscenity
    - Riot
    - Illegal Gambling
    - Racketeering
    - Sexual Assault
    - Rape
    - Statutory Rape
    - Child Pornography
    - Vehicular Manslaughter
    - Hit and Run
    - Insider Trading
    - Piracy (Intellectual Property)
    - Instigation

    Classify the document based on the described crime(s). If multiple crimes are present, identify all relevant crimes. If the document does not fit any listed crimes, respond with 'Unclassified'."


    Return your answer in JSON format with a single key 'crime' and the value being the identified most significant crime.

    Example output:
    jsonCopy{{
    'crime': 'human trafficking'
    }}
    Important notes:

    Always return only the single most significant crime, even if multiple serious crimes are mentioned.
   
    Give the extremely significant crime in the context of the legal document, not just as a standalone term.
    
    Avoid including explanations or additional information in your response.
    Ensure your response is always in valid JSON format.
    Always give a value to the 'crime' key in the JSON output.
    Use single quotes for JSON keys and string values.
    Return the most signficant crime not less significant
    you cant give unclassified as an answer
    You have to give a answer
    Now, analyze the following legal document and extract the single most significant crime:

    Chargesheet:
    {docs}
    """

    prompt = ChatPromptTemplate.from_template(template)

    link_chain = prompt | llm | JsonOutputParser()
    out = link_chain.invoke({"docs": docs})
    print(out)

    link = f"https://r.jina.ai/https://indiankanoon.org/search/?formInput={out['crime']}+doctypes:judgments"
    return link
   


def scrape_jina_ai(url: str) -> str:
    """ "
    get all the text from the url which includes similar cases document link and case heading
    """

    response = requests.get(url)
    return response.text


def get_past_judgement_heading(text_list):
    """
    get similar past judgment heading from the text
    """
    pattern = r"\[(.*?vs.*?on\s+\d{1,2}\s+\w+,\s+\d{4})\]"
    headings = re.findall(pattern, text_list)
    return headings


def past_judgement_link(text_list):
    case_pattern = r"\[Full Document\]\((https://.*?)\)"
    matches = re.findall(case_pattern, text_list)

    docs_link = ["https://r.jina.ai/" + link for link in matches]
    return docs_link


def get_similar_cases_summary(judgement_link, llm):
    """
    gives summary of similar cases from the judgement link
    """

    template = """
    Given the details of the legal document, provide a concise 3-4 line summary with the following format:

    ##### Outcome
    [State the final decision or ruling]

    ##### Reasoning
    [Briefly mention the primary legal reason for the outcome]

    Include only the outcome and its reasoning, without additional details.
    \n\n\n Legal Document : {context}\n\n\n


    The reasoning and outcome should not be more than 3-4 lines

    Make sure the format is followed
  """
    prompt = ChatPromptTemplate.from_template(template)

    sum_chain = prompt | llm | StrOutputParser()

    response = requests.get(judgement_link)
    docs = response.text[:30000]
    token_length = count_tokens(docs)
    if token_length > 131000:
        yield (
            "Document too long. Please refer to the original document at the above link "
        )
        return
    for chunk in sum_chain.stream({"context": docs}):

        yield chunk


def strategy(docs, llm):
    """
    Generates a comprehensive legal strategy for a lawyer based on the provided chargesheet details.


    """

    system = """
   


**Based on the following chargesheet details, generate a comprehensive legal strategy for the lawyer to apply in court. Ensure the strategy is practical, aligned with legal principles, and tailored to the specifics provided. Follow the structure below:**

---

### **Case Overview**
The chargesheet is filed against 10 accused persons for the murder of Ankit Sharma, an Intelligence Bureau officer, during the North-East Delhi riots in February 2020, triggered by protests against the Citizenship Amendment Act (CAA) and National Register of Citizens (NRC). The accused face charges under various sections of the Indian Penal Code (IPC), including murder, rioting, and conspiracy.

### **Legal Analysis**
The investigation reveals the accused, including municipal counselor Tahir Hussain and others, planned and executed the riots. Key evidence includes witness statements, CCTV footage, and forensic reports indicating premeditated actions and connections among the accused. External factors, like the CAA and NRC protests, escalated tensions.

### **Argument Development**
The prosecution will assert that the accused conspired to murder Ankit Sharma, supported by substantial evidence. Conversely, the defense may argue the lack of direct involvement and the circumstantial nature of the evidence, potentially claiming self-defense or provocation.

### **Evidence Strategy**
The prosecution's evidence includes:
1. Witness statements identifying the accused during the riots.
2. CCTV footage linking the accused to the incident.
3. Forensic reports correlating the accused with the crime scene.
4. Call detail records showing coordination among the accused.
5. Interception orders revealing external communications related to the riots.

The defense may challenge evidence admissibility and argue its circumstantial nature, asserting insufficient proof of guilt.

### **Witness Management**
**Prosecution Witnesses:**
1. Eyewitnesses to the riots and murder.
2. Experts on forensic evidence.
3. Police officers detailing investigation procedures.

**Defense Witnesses:**
1. Testimonies negating involvement in the riots.
2. Experts questioning forensic reliability.

### **Risk Mitigation**
**Prosecution Strategies:**
1. Ensure evidence collection adheres to legal standards.
2. Maintain accurate witness identification and statement recording.
3. Protect the rights of the accused during the trial.

**Defense Strategies:**
1. Challenge evidence admissibility.
2. Argue evidence's circumstantial nature and the accused’s self-defense claims.

### **Courtroom Tactics**
**Prosecution Tactics:**
1. Create a clear narrative of events leading to the murder.
2. Use visual aids (CCTV, diagrams) for evidence clarity.
3. Employ expert witnesses to reinforce the case.

**Defense Tactics:**
1. Challenge evidence and its admissibility.
2. Highlight weaknesses in prosecution arguments.

### **Conclusion**
The prosecution will rely on robust evidence, including witness accounts and forensic data, to establish the guilt of the accused. The defense will focus on evidentiary challenges and claims of self-defense. The court's decision will be based on the presented evidence and arguments.



   """

    human = "{docs}"
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])

    strategy_chain = prompt | llm | StrOutputParser()
    for chunk in strategy_chain.stream({"docs": docs}):

        yield chunk


def extract_text_from_pdf(pdf_file):
    """
    Extracts the text from a given PDF file.


    """

    text = ""
    pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        text += page.get_text()
    return text, pdf_document


def raptor_retriever(docs_text: str, index_name: str):
    """
    Retrieves a retriever object

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

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    return retriever


def raptor(retriever, llm, question: str):
    """
    this is the rag system generation and augmentation step
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
