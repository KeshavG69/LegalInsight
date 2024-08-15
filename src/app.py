import streamlit as st


import streamlit_ext as ste
from helper import *
# from raptor import *

print("Refreshing")


st.set_page_config(layout="wide")

st.title("Legal Document Assistant")

uploaded_file = ste.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    document_text = extract_text_from_pdf(uploaded_file)


    col1, col2,col3 = st.columns(3)

    with col1:
        st.header("Document Summary")
        if 'summary_text' not in st.session_state:
            st.session_state.summary_text =st.write_stream(get_summary(document_text))
        else:
            st.write(st.session_state.summary_text)

        st.write('------------------------------------------------------------------')
        ste.download_button(label="Download Summary....", data=st.session_state.summary_text,file_name='summary.txt')

    with col2:
        st.header("Previous Judgments")
        if 'judgment_text' not in st.session_state:
            st.session_state.judgment_text=''

            link='https://indiankanoon.org/search/?formInput=Murder+doctypes:judgments'
            print(link)
            questions=questions_web_search(scrape_jina_ai(link))[:1]

            for i,q in enumerate(questions):

                st.markdown(f'### Past Case Number: {i+1}')
                st.session_state.judgment_text+=f'### Past Case Number: {i+1}'
                print('hello')
                st.session_state.judgment_text+=(st.write_stream(get_similar_cases_summary(q)))
                print('hello1')
                st.write('------------------------------------------------------------------')
                st.session_state.judgment_text+='------------------------------------------------------------------'
                print('hello2')
                st.session_state.judgment_text+=('\n\n\n')
                print('hello3')

        else:

            st.write(st.session_state.judgment_text)

            st.write('------------------------------------------------------------------')
        ste.download_button(label="Download Previous Judgements....", data=''.join([str(element) for element in st.session_state.judgment_text]),file_name='previous_judgement.txt')

    with col3:
        st.header("Strategy")
        if 'strategy_text' not in st.session_state:
            st.session_state.strategy_text=st.write_stream(strategy(document_text))
        else:
            st.write(st.session_state.strategy_text)

        st.write('------------------------------------------------------------------')
        ste.download_button(label="Download Strategy....", data=st.session_state.strategy_text,file_name='strategy.txt')

    st.header("Chat with PDF")
    index_name = st.text_input(
        "Enter The Project Name(Always give unique project names)"
    )

    if index_name.strip().lower():
        retriever = raptor_retriever(document_text, index_name)
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        for message in st.session_state.messages:
            with st.chat_message(message['role']):
                st.markdown(message['content'])
        user_question = st.text_input("Ask a question about the document:")
        if user_question:
            st.session_state.messages.append({'role':'user','content':user_question})
            with st.chat_message('user'):
                st.markdown(user_question)
            
            with st.chat_message('assistant'):
                response=st.write_stream(raptor(retriever, user_question))

            st.session_state.messages.append({'role':'assistant','content':response})



























