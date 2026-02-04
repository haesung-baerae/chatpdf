__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# 임베딩
from langchain_ollama import OllamaEmbeddings

# 벡터 저장
from langchain_chroma import Chroma

# 검색기
# https://wikidocs.net/231603참고
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

import logging


for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("langchain.retrievers.multi_query")
logger.setLevel(logging.INFO)

load_dotenv()


import streamlit as st
import os

st.title("ChatPDF")
st.write("---")
# 첫 번째 파라미터: 파일 업로드 필도의 레이블, 필수
# 두 번째 파라미터: 업로드를 허용할 파일 형식, 지정하지 않으면 모든 파일 형식 업로드 가능
uploaded_file = st.file_uploader("PDF파일을 올려주세요!", type=['pdf'])
st.write("---")

import tempfile
def pdf_to_document(uploaded_file):
    temp_dir = tempfile.TemporaryDirectory()
    temp_filepath = os.path.join(temp_dir.name, uploaded_file.name)
    with open(temp_filepath, 'wb') as f:
        f.write(uploaded_file.getvalue())
    #loader = PyPDFLoader(temp_filepath)
    loader = PyMuPDFLoader(temp_filepath)
    pages = loader.load_and_split()
    return pages
if uploaded_file is not None:
    pages = pdf_to_document(uploaded_file)

    # chunk_size = 각 청크의 최대 길이, 글자 단위
    # chunk_overlap = 인접한 청크 사이에 중복되는 영역, 깔끔하게 끝나면 겹치지 않음
    # length_function = 청크 길이를 측정하는 함수
    # is_separator_regex : True- 정규표현식을 통해 구분자 처리, False- 구분자를 문자열로 해석
    text_splitter= RecursiveCharacterTextSplitter(
        chunk_size = 500, #800
        chunk_overlap = 100, #100
        length_function=len,
        is_separator_regex = False,
        separators=[
        "\n제",           # '제1조', '제2조' 등 조항 시작점 기준
        "\n\s*제\d+조",   # 정규표현식 지원 시: 숫자 포함 조항 기준
        "\n\n",           # 문단 구분
        "\n",             # 줄바꿈
        " ",              # 띄어쓰기
        ""                # 글자 단위
    ]
        
    )
    texts = text_splitter.split_documents(pages)

    # 임베딩
    #embeddings_model = OllamaEmbeddings(model='mxbai-embed-large')
    from langchain_openai import OpenAIEmbeddings
    embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

    # 벡터 저장
    file_stem = os.path.splitext(uploaded_file.name)[0]
    persist_db_path = os.path.join('./embedding_db', file_stem)
    if os.path.exists(persist_db_path):
        print("기존 임베딩 데이터를 불러옵니다..")
        db = Chroma(persist_directory=persist_db_path,
                    embedding_function=embeddings_model)
    else :
        print("새롭게 임베딩을 진행합니다..")
        db = Chroma.from_documents(texts, embeddings_model,
                            persist_directory=persist_db_path)
        
    
    st.header("PDF에게 질문해보세요!")    
    question = st.text_input("질문을 입력하세요")
        
    if st.button("질문하기"):
        with st.spinner("Wait for it..."):
                        
            # 체인을 돌리기 전, 리트리버만 따로 실행해서 결과 확인
            # llm = ChatOpenAI(temperature = 0)
            # retriever_from_llm = MultiQueryRetriever.from_llm(retriever=db.as_retriever(search_kwargs={'k':4}), llm=llm)
            # docs = retriever_from_llm.invoke("당직근무 제외 조건이 뭐야?")
            # print(f"가져온 문서 개수: {len(docs)}")
            # for i, doc in enumerate(docs):
            #     print(f"--- 문서 {i+1} (페이지: {doc.metadata.get('page')}) ---")
            #     print(doc.page_content[:200]) # 가져온 내용의 앞부분 확인
            # 답변 생성
            llm = ChatOpenAI(temperature = 0)
            retriever_from_llm = MultiQueryRetriever.from_llm(
                retriever=db.as_retriever(search_kwargs={'k':4}), 
                llm=llm)

            # 프롬프트 템플릿 
            # 방법1 프롬프트 허브 활용
            #from langchain_classic import hub
            #prompt = hub.pull('rlm/rag-prompt')

            # 방법2 
            # 프롬프트에 따라 page 번호가 다르게 나올 수 있음... 중요하다!
            from langchain_core.prompts import ChatPromptTemplate
            template = """당신은 부대관리훈령에 대해 답변하는 전문 보좌관입니다.
            아래 제공된 문맥(Context)을 바탕으로 질문에 답하세요.

            [답변 가이드라인]
            1. 반드시 항목별로 번호를 매겨(1., 2., 3...) 일목요연하게 정리하여 답변하세요.
            2. 각 항목 뒤에는 반드시 '[[문서 출처: n페이지]]' 형태로 출처를 붙이세요.
            3. 문서 본문에 명시된 '제O조' 조항 번호를 찾을 수 있다면 반드시 포함하세요.
            4. 본문에 없는 내용을 추측하거나, 인접한 다른 조항의 내용을 섞어서 답변하지 마세요.
            5. '영내 순찰' 등 당직자의 임무와 '면제 조건'을 명확히 구분하여 '면제'에 해당하는 내용만 답변하세요.
            
            [답변 구성 원칙]
            1. 각 항목은 독립적인 문장으로 구성하세요. 
            2. 서로 다른 조항이나 페이지에서 온 문장을 하나로 합쳐서 새로운 인과관계(예: ~하면 ~된다)를 만들지 마세요.
            3. '제외된다', '면제된다'라는 결론이 본문에 명시된 내용만 '제외 조건'으로 분류하세요.
            4. 문단이 바뀌거나 페이지가 바뀌어 문맥이 끊기는 내용은 서로 별개의 항목으로 취급하세요.

            [예외 처리]
            - 질문과 관련된 키워드는 있지만 구체적 정보가 없으면: "00에 대한 언급은 있으나, 구체적인 규정은 찾을 수 없습니다."라고 답하세요.
            - 관련 내용이 아예 없으면: "해당 조항을 찾을 수 없습니다."라고 답하세요.

            문맥 : {context}
            질문 : {question}
            답변 : """
            prompt = ChatPromptTemplate.from_template(template)

            # 생성
            # RunnablePassthrough : 사용자 입력을 llm체인에 그대로 전달
            from langchain_core.runnables import RunnablePassthrough
            from langchain_core.output_parsers import StrOutputParser
            # 1단계
            #def format_docs(docs):
            #    return '\n\n'.join(doc.page_content for doc in docs)

            # 2단계
            # 문서 조각들을 하나의 문자열로 합치되, 페이지 번호를 포함시키는 함수
            import re
            def format_docs(docs):
                formatted = []
                for doc in docs:
                    page_num=doc.metadata.get('page',0)+1
                    content = doc.page_content
                    content = re.sub(r'법제처\s+\d+\s+국가법령정보센터','',content)
                    content = re.sub(r'부대관리훈령','',content)
                    content = re.sub(r'\n+', '\n', content).strip()
                    formatted_doc = f"[[문서 출처:{page_num}페이지]]\n{content}"                    
                    formatted.append(formatted_doc)
                return '\n\n'.join(formatted)


            rag_chain = (
                {'context': retriever_from_llm|format_docs, 'question':RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )

            # 질문
            #result = rag_chain.invoke("부대관리훈령에서 정의하는 '병영생활'의 범위는어디까지야?")
            #result = rag_chain.invoke("평일과 휴일의 병사 휴대전화 사용 가능 시간을 각각 알려줘")
            #result = rag_chain.invoke("도박하면 처벌이 어떻게 돼?")
            result = rag_chain.invoke(question)
            print("="*50)
            print(result)
            st.write(result)