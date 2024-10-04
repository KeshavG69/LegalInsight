from helper import *


# os.environ['LANGCHAIN_TRACING_V2'] = 'true'
# os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
# os.environ['LANGCHAIN_API_KEY'] = userdata.get('LANGCHAIN_API_KEY')

st.set_page_config(layout="wide")

st.title("Legal Document Assistant")

uploaded_file = ste.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    document_text, link_text = extract_text_from_pdf(uploaded_file)
    print(link_text)
    print(type(link_text))

    col1, col2, col3 = st.columns(3)

    with col1:
        st.header("Document Summary")
        if "summary_text" not in st.session_state:
            st.session_state.summary_text = st.write_stream(
                get_summary(document_text, local_llm)
            )
        else:
            st.write(st.session_state.summary_text)

        st.write("------------------------------------------------------------------")
        ste.download_button(
            label="Download Summary....",
            data=st.session_state.summary_text,
            file_name="summary.txt",
        )

    with col2:
        st.header("Previous Similar Judgments")
        if "judgment_text" not in st.session_state:
            st.session_state.judgment_text = ""
            link=get_link(link_text,local_llm_json)
            print(link)
            indian_kanoon_text = scrape_jina_ai(link)
            

            past_judgement_links = past_judgement_link(indian_kanoon_text)
            print(past_judgement_links)

            past_judgement_heading = get_past_judgement_heading(indian_kanoon_text)
            print(past_judgement_heading)
            for i, q in enumerate(past_judgement_links):

                st.markdown(f"### Past Case Number: {i+1}")
                st.session_state.judgment_text += f"### Past Case Number: {i+1}"
                st.markdown(f"### Case Heading: {past_judgement_heading[i]}")
                st.session_state.judgment_text += (
                    f"### Case Heading: {past_judgement_heading[i]}"
                )

                st.markdown(f'### Doc Link: {q.replace("https://r.jina.ai/", "")}')
                st.session_state.judgment_text += f'### Doc Link: {q.replace("https://r.jina.ai/", "")}'

                st.session_state.judgment_text += st.write_stream(
                    get_similar_cases_summary(q, llm)
                )

                st.write(
                    "------------------------------------------------------------------"
                )
                st.session_state.judgment_text += (
                    "------------------------------------------------------------------"
                )

                st.session_state.judgment_text += "\n\n\n"

        else:

            st.write(st.session_state.judgment_text)

            st.write(
                "------------------------------------------------------------------"
            )
        ste.download_button(
            label="Download Previous Judgements....",
            data="".join([str(element) for element in st.session_state.judgment_text]),
            file_name="previous_judgement.txt",
        )

    with col3:
        st.header("Strategy")
        if "strategy_text" not in st.session_state:
            st.session_state.strategy_text = st.write_stream(
                strategy(document_text, local_llm)
            )
        else:
            st.write(st.session_state.strategy_text)

        st.write("------------------------------------------------------------------")
        ste.download_button(
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
            retriever = raptor_retriever(document_text, index_name)
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
                response = st.write_stream(raptor(retriever, local_llm, user_question))

            st.session_state.messages.append({"role": "assistant", "content": response})
