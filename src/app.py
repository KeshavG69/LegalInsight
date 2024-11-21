__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')



from helper import *


st.set_page_config(layout="wide")

st.title("Legal Document Assistant")

uploaded_file = ste.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    document_text = extract_text_from_pdf(uploaded_file)

    tabs = st.tabs(["Document Summary", "Previous Similar Judgments", "Strategy"])
    
    with tabs[0]:  # Document Summary tab
        if "summary_text" not in st.session_state:
            st.session_state.summary_text = st.write_stream(
                get_summary(document_text)
            )
        else:
            st.write(st.session_state.summary_text)
    
        st.write("------------------------------------------------------------------")
        st.download_button(
            label="Download Summary....",
            data=st.session_state.summary_text,
            file_name="summary.txt",
        )
    with tabs[1]:  # Previous Similar Judgments tab
        if "judgment_text" not in st.session_state:
            st.session_state.judgment_text = ""
            link = get_link(document_text)
            print(link)
            indian_kanoon_text = scrape_jina_ai(link)
    
            past_judgement_links = past_judgement_link(indian_kanoon_text)
            past_judgement_heading = get_past_judgement_heading(indian_kanoon_text)
    
            for i, q in enumerate(past_judgement_links):
                st.markdown(f"### Past Case Number: {i + 1}")
                st.session_state.judgment_text += f"### Past Case Number: {i + 1}"
                st.markdown(f"### Case Heading: {past_judgement_heading[i]}")
                st.session_state.judgment_text += (
                    f"### Case Heading: {past_judgement_heading[i]}"
                )
    
                st.markdown(f'### Doc Link: {q.replace("https://r.jina.ai/", "")}')
                st.session_state.judgment_text += f'### Doc Link: {q.replace("https://r.jina.ai/", "")}'
    
                st.session_state.judgment_text += st.write_stream(
                    get_similar_cases_summary(q)
                )
    
                st.write("------------------------------------------------------------------")
                st.session_state.judgment_text += (
                    "------------------------------------------------------------------"
                )
    
                st.session_state.judgment_text += "\n\n\n"
    
        else:
            st.write(st.session_state.judgment_text)
            st.write("------------------------------------------------------------------")
    
        st.download_button(
            label="Download Previous Judgements....",
            data="".join([str(element) for element in st.session_state.judgment_text]),
            file_name="previous_judgement.txt",
        )
    with tabs[2]:  # Strategy tab
        if "strategy_text" not in st.session_state:
            st.session_state.strategy_text = st.write_stream(
                strategy(document_text)
            )
        else:
            st.write(st.session_state.strategy_text)
    
        st.write("------------------------------------------------------------------")
        st.download_button(
            label="Download Strategy....",
            data=st.session_state.strategy_text,
            file_name="strategy.txt",
        )

    st.header("Chat with PDF")
    index_name = (
        st.text_input("Enter The Project Name(Always give unique project names)")
        .strip()
        .lower()
    )

    if index_name:
        with st.spinner(f"Opening New Project {index_name}"):
            retriever = raptor_retriever_pinecone(document_text, index_name)
        if "messages" not in st.session_state:
            st.session_state.messages = []
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        user_question = st.text_input("Ask a question about the document:")
        if user_question:
            st.session_state.messages.append({"role": "user", "content": user_question})
            with st.chat_message("user"):
                st.markdown(user_question)

            with st.chat_message("assistant"):
                response = st.write_stream(raptor(retriever, user_question))

            st.session_state.messages.append({"role": "assistant", "content": response})
