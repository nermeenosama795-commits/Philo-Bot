import streamlit as st
import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# --- Step 2: Custom CSS ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=Lora:ital,wght@0,400;0,700;1,400&display=swap');

/* 1. تغيير الخلفية للون الكريمي (البيج) */
.stApp {
    background-color: #F2EFE9 !important; /* لون ورق البردي الهادئ */
}

/* 2. تنسيق الخط العام (استخدام خط Lora المريح للقراءة) */
html, body, [class*="css"], .stMarkdown {
    font-family: 'Lora', serif !important;
    color: #1A1A1A !important; /* أسود فحمي مريح للعين */
}

/* 3. تنسيق الهيدر (العنوان) ليظهر مثل الصورة */
.hero {
    text-align: center;
    padding: 2rem 0;
    margin-bottom: 2rem;
}

.hero h1 {
    font-family: 'Playfair Display', serif; /* خط كلاسيكي للعناوين */
    font-size: 4rem;
    letter-spacing: 0.3rem;
    color: #000000 !important;
    text-transform: uppercase; /* حروف كبيرة */
    margin-bottom: 0;
}

.hero p {
    font-size: 1rem;
    letter-spacing: 0.1rem;
    color: #4A4A4A !important;
    text-transform: uppercase;
}

/* 4. تنسيق فقاعات الدردشة (مثل الصورة) */
.stChatMessage {
    background-color: #FAF9F6 !important; /* خلفية افتح قليلاً للرسائل */
    border: 1px solid #D1CDC4 !important; /* إطار خفيف جداً */
    border-radius: 5px !important; /* حواف حادة قليلاً لتعطي طابع الكتب */
    box-shadow: 2px 2px 5px rgba(0,0,0,0.05);
    margin-bottom: 1rem;
}

/* 5. تنسيق صندوق الإدخال (الذي يكتب فيه المستخدم) */
.stChatInput textarea {
    background-color: #FFFFFF !important;
    border: 1px solid #D1CDC4 !important;
    color: #000000 !important;
    border-radius: 5px !important;
}

/* إخفاء العناصر الزائدة لزيادة نظافة التصميم */
#MainMenu, footer, header { visibility: hidden; }
</style>

<div class="hero">
    <h1>PHILO</h1>
    <p>Your Philosophical Companion</p>
</div>
""", unsafe_allow_html=True)

# --- Step 3: Build the RAG Chain ---
@st.cache_resource
def build_rag_chain():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vector_store = FAISS.load_local("faiss_philosophy_index", embeddings, allow_dangerous_deserialization=True)

    retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 4})

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.2,
        google_api_key="AIzaSyC5bSQMEqNer6E27QUbDTq-9kHW-UL35Rc"
    )

    # ✅ Fix 1: these were accidentally outside the function — now properly indented
    system_prompt = (
        "أنت الآن خبير متخصص في الفلسفة الكلاسيكية (أفلاطون، أرسطو، والمدرسة الرواقية).\n"
        "مهمتك هي الإجابة على أسئلة المستخدم بناءً على النصوص المسترجعة فقط من الكتب المرفقة.\n"
        "يجب أن يكون أسلوبك في الإجابة حكيماً، هادئاً، ومنظماً.\n"
        "إذا كانت المعلومة المطلوبة غير موجودة في السياق المرفق، قل بوضوح: 'عذراً، هذه المعلومة ليست متوفرة في المصادر الفلسفية المتاحة لدي' ولا تقم بتأليف إجابة من عندك.\n"
        "عند الإجابة، حاول دائماً نسب الأفكار لصاحبها (مثلاً: وفقاً لأرسطو.. أو كما ذكر أفلاطون).\n"
        "\n\n"
        "Context: {context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    # ✅ Fix 2: these had 8-space indent — corrected to 4-space (inside function)
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, combine_docs_chain)

# --- Step 4: Chat History ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Step 5: Run the App ---
rag_chain = build_rag_chain()

# عرض الرسائل السابقة
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# صندوق الإدخال
query = st.chat_input("اسأل عن كتاب، مؤلف، أو موضوع...")

if query:
    # حفظ وعرض رسالة المستخدم
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)

    with st.spinner("جاري البحث في الكتب..."):
        result = rag_chain.invoke({"input": query})
        answer = result["answer"]

    # حفظ وعرض رد البوت
    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.write(answer)

        # عرض المصادر (Expander) كما في السلايد الأخير
        with st.expander("عرض المصادر المقتبسة"):
            for i, doc in enumerate(result["context"], 1):
                st.caption(f"الجزء رقم {i}")
                st.write(doc.page_content)
                st.divider()
