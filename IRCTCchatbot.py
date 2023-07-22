import streamlit as st
from streamlit_chat import message
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
import tempfile
import openai
from langchain.chat_models import ChatOpenAI
import os
import openai


os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"
API_KEY = "YOUR_API_KEY"

uploaded_file = st.sidebar.file_uploader("upload", type="csv")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    loader = CSVLoader(file_path=tmp_file_path, encoding="utf-8", csv_args={
        'delimiter': ','})
    data = loader.load()


embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(data, embeddings)

chain = ConversationalRetrievalChain.from_llm(
    llm=ChatOpenAI(temperature=0.0, model_name='gpt-3.5-turbo'),
    retriever=vectorstore.as_retriever())

text = str()
text = str()
text = text
source_station = str()
destination_station = str()


def conversational_chat(query):

    openai.api_key = "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"

    prompt = (
        """
        Language Model: NER with Langchain
        User: Please identify the source station and destination station from the user query.
        Always put the names of source station and destination station in double inverted commas.
        query: "{query}" 
        Language Model:
        
        """
    )

    def complete(prompt):
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=100,
            temperature=0.5,
            n=1,
            stop=None
        )
        return response.choices[0].text.strip()
    # Define the NER function

    def perform_ner(query):
        full_prompt = f'{prompt}Perform NER on "{query}"\n'
        response = complete(full_prompt)
        return response

    output = perform_ner(query)

    import re

    retry_count = 0
    max_retries = 5
    while retry_count < max_retries:
        output = perform_ner(query)
    #     print(output)

        # Extract the source station and destination station using regular expressions
        matches = re.findall(r'"([^"]+)"', output)

        if len(matches) >= 2:
            source_station = matches[0]
            destination_station = matches[1]

            # st.write("Source Station:", source_station)
            # st.write("Destination Station:", destination_station)
            break
        else:
            retry_count += 1
    #         print("Retry:", retry_count)

    if retry_count == max_retries:
        st.write("Extraction failed after maximum retries.")

    import pandas as pd
    import networkx as nx

    # Read the CSV file containing the train data
    train_data = pd.read_csv('finalds.csv')

    # Create a directed graph
    G = nx.DiGraph()

    # Add edges to the graph based on the train data
    for _, row in train_data.iterrows():
        source = row['Source Station']
        destination = row['Destination Station']
        train_name = row['Train Name']
        distance = row['Distance']
        G.add_edge(source, destination, train=train_name, distance=distance)

        # Check if the source_station and destination_station exist in the graph
    if source_station not in G.nodes:
        st.write("Source station not found in the train data.")
        exit()
    if destination_station not in G.nodes:
        st.write("Destination station not found in the train data.")
        exit()

    # Find the shortest path between the source_station and destination_station
    try:
        shortest_path = nx.shortest_path(
            G, source_station, destination_station, weight='distance')
    except nx.NetworkXNoPath:
        st.write("No path found between the source and destination stations.")
        exit()

    # Get the train names along the shortest path
    train_names = [G.edges[edge]['train']
                   for edge in zip(shortest_path[:-1], shortest_path[1:])]

    # Print the train names
    # print("Trains from", source_station, "to", destination_station, "via the shortest path:")
    for train_name in train_names:
        #   print(train_name)
        # Define the custom prompt
        prompt_2 = """
        Language Model:
        Your task is to perform the following action:
        Just give the name of the trains in variable {train_name} that can be boarded to reach the destination in shortest way in a friendly language style sentence.
        
        """

        # Define the completion function
        def complete_prompt2(prompt_2, train_name):
            full_prompt = prompt_2.format(train_name=train_name)
            response = openai.Completion.create(
                engine="text-davinci-003",
                prompt=full_prompt,
                max_tokens=100,
                temperature=0.5,
                n=1,
                stop=None
            )
            return response.choices[0].text.strip()

        # Example usage
    # Convert the train names to a comma-separated string
    train_name = ", ".join(train_names)
    output2 = train_name

    output3 = complete_prompt2(prompt_2, train_name)

    # Print the output
    result = (output3)

    # return result

    result = chain({
        "question": query,
        "chat_history": st.session_state['history'],
        "output": output3})
    st.session_state['history'].append(text)
    st.session_state['history'].append(output3)
    return (result["output"])


# In[ ]:
if 'history' not in st.session_state:
    st.session_state['history'] = []

if 'generated' not in st.session_state:
    st.session_state['generated'] = [
        "Hello ! Ask me anything about " + uploaded_file.name + " 🤗"]

if 'past' not in st.session_state:
    st.session_state['past'] = ["Hey ! 👋"]

# container for the chat history
response_container = st.container()
# container for the user's text input
container = st.container()


with container:
    with st.form(key='my_form', clear_on_submit=True):

        query = st.text_input(
            "text:", placeholder="Talk about your csv data here (:", key='input')

        # text=text
        submit_button = st.form_submit_button(label='Send')

    if submit_button and query:
        output4 = conversational_chat(query)

        st.session_state['past'].append(query)
        st.session_state['generated'].append(output4)

if st.session_state['generated']:
    with response_container:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state["past"][i], is_user=True, key=str(
                i) + '_user', avatar_style="big-smile")
            message(st.session_state["generated"][i],
                    key=str(i), avatar_style="thumbs")
