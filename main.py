import time

import streamlit as st
from transformers.pipelines import pipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


st.success('Gratulacje! Z powodzeniem uruchomiłeś aplikację')

st.spinner()
with st.spinner(text='Ładowanie...'):
    time.sleep(2)
    st.success('Ładowanie zakończone')

st.title('Tłumacz')
st.write('Proszę wybrać jedną z dwóch opcji:')
st.write('1. Pozwala sprawdzać wydźwięk emocjonalny tekstu napisanego w języku angielskim.')
st.write('2. Pozwala na tłumaczenie tekstów z języka angielskiego na język niemiecki')

option_list = [
    'Wydźwięk emocjonalny tekstu (eng)',
    'Tłumacz z języka angielskiego na niemiecki'
]

option = st.selectbox(
    'Opcje',
    option_list
)

if option == option_list[0]:
    text = st.text_area(label='Proszę podać tekst do analizy')
    if text:
        try:
            classifier = pipeline("sentiment-analysis", model='cardiffnlp/twitter-roberta-base-sentiment-latest')
            answer = classifier(text)
            st.spinner()

            st.write(answer[0]['label'])
        except Exception as e:
            st.error('Wystąpił błąd proszę spróbować ponownie')
            print(str(e))

elif option == option_list[1]:
    text_to_translate = st.text_area(label='Tekst do przetłumaczenia')
    if text_to_translate:
        try:
            AutoTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-de')
            AutoModelForSeq2SeqLM.from_pretrained('Helsinki-NLP/opus-mt-en-de')

            translator = pipeline('translation_en_to_de', model='Helsinki-NLP/opus-mt-en-de')
            answer = translator(text_to_translate, max_length=100)[0]['translation_text']

            st.spinner()
            with st.spinner(text='Tłumaczenie tesktu w toku'):
                time.sleep(2)
                st.success(answer)

        except Exception as e:
            st.error('Wystąpił błąd proszę spróbować ponownie')
            print(str(e))

st.write('Autor: Jakub Wiańczyk, Indeks: s18825')