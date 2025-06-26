# Import Libraries and Modules
from dotenv import load_dotenv
import os
# import json
# import datetime
# from typing import List, Dict, Any

import streamlit as st
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory, ConversationSummaryBufferMemory
from langchain.chains import ConversationChain
# from langchain.schema import BaseMessage, HumanMessage, AIMessage
# import pandas as pd

# Load environment variables
load_dotenv()

# Configuration
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

# Available models
AVAILABLE_MODELS = {
    "gemma2-9b-it": "Gemma 2 9B - Balanced performance and speed",
    "deepseek-r1-distill-llama-70b": "DeepSeek R1 - Advanced reasoning capabilities",
    "llama3-8b-8192": "Llama 3 8B - Fast and efficient",
    "meta-llama/llama-prompt-guard-2-22m": "Llama Prompt Guard - Safety focused"
}

# Memory types
MEMORY_TYPES = {
    "Buffer": ConversationBufferMemory,
    "Summary Buffer": ConversationSummaryBufferMemory
}

# Initialize session states
def initialize_session_state():
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(return_messages=True)
    
    if "history" not in st.session_state:
        st.session_state.history = []
    
    if "chat_sessions" not in st.session_state:
        st.session_state.chat_sessions = {"Default": []}
    
    if "current_session" not in st.session_state:
        st.session_state.current_session = "Default"
    
    if "system_prompt" not in st.session_state:
        st.session_state.system_prompt = "You are a helpful AI assistant. Be concise and informative."
    
    if "message_count" not in st.session_state:
        st.session_state.message_count = 0

def setup_page():
    """Setup Streamlit page configuration"""
    st.set_page_config(
        page_title="AskBot Conversational Chatbot",
        page_icon="💬",
        layout="centered",
        initial_sidebar_state="expanded"
    )

def create_sidebar():
    st.sidebar.title("AskBot Settings")
    
    # API Key Status
    if GROQ_API_KEY:
        st.sidebar.success("✅ GROQ API Key Loaded")
    else:
        st.sidebar.error("GROQ API Key Missing")
        st.sidebar.info("Please add your GROQ_API_KEY to your .env file")
    
    st.sidebar.divider()
    
    # Model Selection
    st.sidebar.subheader("🧠 Model Configuration")
    model_name = st.sidebar.selectbox(
        "Select Model",
        options=list(AVAILABLE_MODELS.keys()),
        format_func=lambda x: AVAILABLE_MODELS[x],
        help="Choose the AI model for conversations"
    )
    
    temperature = st.sidebar.slider(
        "Temperature",
        min_value=0.00,
        max_value=2.00,
        value=0.7,
        step=0.1,
        help="Controls randomness: lower = more focused, higher = more creative"
    )
    
    max_tokens = st.sidebar.slider(
        "Max Tokens",
        min_value=50,
        max_value=2000,
        value=500,
        step=50,
        help="Maximum length of AI responses"
    )
    
    st.sidebar.divider()
    
    # Memory Configuration
    st.sidebar.subheader("🧠 Memory Settings")
    memory_type = st.sidebar.selectbox(
        "Memory Type",
        options=list(MEMORY_TYPES.keys()),
        help="Choose how the bot remembers conversation history"
    )
    
    if memory_type == "Summary Buffer":
        max_token_limit = st.sidebar.slider(
            "Memory Token Limit",
            min_value=100,
            max_value=2000,
            value=1000,
            help="When to start summarizing old messages"
        )
    else:
        max_token_limit = None
    
    st.sidebar.divider()
    
    # System Prompt
    st.sidebar.subheader("📝 System Prompt")
    system_prompt = st.sidebar.text_area(
        "Customize AI Behavior",
        value=st.session_state.system_prompt,
        height=100,
        help="Define how the AI should behave"
    )
    
    if st.sidebar.button("Update System Prompt"):
        st.session_state.system_prompt = system_prompt
        st.sidebar.success("System prompt updated!")
    
    return model_name, temperature, max_tokens, memory_type, max_token_limit

def create_chat_session_manager():
    # Create chat session management interface
    st.sidebar.divider()
    st.sidebar.subheader("💬 Chat Sessions")
    
    # Current session selector
    current_session = st.sidebar.selectbox(
        "Current Session",
        options=list(st.session_state.chat_sessions.keys()),
        index=list(st.session_state.chat_sessions.keys()).index(st.session_state.current_session)
    )
    
    # New session creation
    col1, col2 = st.sidebar.columns(2)
    with col1:
        new_session_name = st.text_input("New Session", placeholder="Session name")
    with col2:
        if st.button("Create") and new_session_name:
            if new_session_name not in st.session_state.chat_sessions:
                st.session_state.chat_sessions[new_session_name] = []
                st.session_state.current_session = new_session_name
                st.rerun()
    
    # Switch session
    if current_session != st.session_state.current_session:
        st.session_state.current_session = current_session
        st.session_state.history = st.session_state.chat_sessions[current_session].copy()
        # Reset memory for new session
        st.session_state.memory = ConversationBufferMemory(return_messages=True)
        st.rerun()
    
    # Session management buttons
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("Clear Session"):
            st.session_state.chat_sessions[st.session_state.current_session] = []
            st.session_state.history = []
            st.session_state.memory = ConversationBufferMemory(return_messages=True)
            st.rerun()
    
    with col2:
        if st.button("Delete Session") and len(st.session_state.chat_sessions) > 1:
            del st.session_state.chat_sessions[st.session_state.current_session]
            st.session_state.current_session = list(st.session_state.chat_sessions.keys())[0]
            st.session_state.history = st.session_state.chat_sessions[st.session_state.current_session].copy()
            st.rerun()


def create_llm(model_name: str, temperature: float, max_tokens: int):
    # initialize llm
    try:
        return ChatGroq(
            groq_api_key=GROQ_API_KEY,
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            streaming=True
        )
    except Exception as e:
        st.error(f"Error initializing model: {str(e)}")
        return None

def create_memory(memory_type: str, max_token_limit: int = None):
    # Memory type selection
    if memory_type == "Summary Buffer" and max_token_limit:
        return ConversationSummaryBufferMemory(
            return_messages=True,
            max_token_limit=max_token_limit
        )
    else:
        return ConversationBufferMemory(return_messages=True)

def display_chat_statistics():
    with st.expander("📊 Chat Statistics"):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Messages", len(st.session_state.history))
        
        with col2:
            user_messages = len([msg for role, msg in st.session_state.history if role == "user"])
            st.metric("User Messages", user_messages)
        
        with col3:
            ai_messages = len([msg for role, msg in st.session_state.history if role == "assistant"])
            st.metric("AI Messages", ai_messages)
        
        with col4:
            total_chars = sum(len(msg) for _, msg in st.session_state.history)
            st.metric("Total Characters", total_chars)

def main():
    setup_page()
    initialize_session_state()
    
    st.title("💬 AskBot - Ask Anything")
    st.markdown("*Powered by Groq and LangChain*")
    
    # Sidebar configuration
    model_name, temperature, max_tokens, memory_type, max_token_limit = create_sidebar()
    create_chat_session_manager()
    
    # Check API key
    if not GROQ_API_KEY:
        st.error("🔑 GROQ API Key is required. Please add it to your .env file.")
        st.stop()
    
    # Chat statistics
    display_chat_statistics()
    
    # Chat input
    user_input = st.chat_input("Type your message here...")
    
    if user_input:
        # Add user message to history
        st.session_state.history.append(("user", user_input))
        st.session_state.chat_sessions[st.session_state.current_session].append(("user", user_input))
        
        # Create LLM and memory
        llm = create_llm(model_name, temperature, max_tokens)
        
        if llm is None:
            st.error("Failed to initialize the language model. Please check your configuration.")
            st.stop()
        
        # Update memory type if changed
        current_memory_type = type(st.session_state.memory).__name__
        if (memory_type == "Buffer" and "Buffer" not in current_memory_type) or \
           (memory_type == "Summary Buffer" and "Summary" not in current_memory_type):
            st.session_state.memory = create_memory(memory_type, max_token_limit)
        
        # Create conversation chain
        try:
            conversation = ConversationChain(
                llm=llm,
                memory=st.session_state.memory,
                verbose=False
            )
            
            # Add system prompt context
            full_input = f"System: {st.session_state.system_prompt}\n\nUser: {user_input}"
            
            # Get AI response with streaming
            with st.spinner("🤔 Thinking..."):
                ai_response = conversation.predict(input=full_input)
            
            # Add AI response to history
            st.session_state.history.append(("assistant", ai_response))
            st.session_state.chat_sessions[st.session_state.current_session].append(("assistant", ai_response))
            st.session_state.message_count += 1
            
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
            ai_response = "I apologize, but I encountered an error while processing your request. Please try again."
            st.session_state.history.append(("assistant", ai_response))
    
    # Display chat history
    for i, (role, message) in enumerate(st.session_state.history):
        with st.chat_message(role):
            st.markdown(message)
            
            # Add copy button for assistant messages
            if role == "assistant":
                if st.button(f"📋 Copy", key=f"copy_{i}"):
                    st.code(message)

    # Quick actions
    if st.session_state.history:
        st.divider()
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 Regenerate Response"):
                if len(st.session_state.history) >= 2 and st.session_state.history[-1][0] == "assistant":
                    # Remove last AI response from both history and chat sessions
                    st.session_state.history.pop()
                    st.session_state.chat_sessions[st.session_state.current_session].pop()
                    
                    # Get the last user message
                    last_user_message = st.session_state.history[-1][1] if st.session_state.history else ""
                    
                    if last_user_message:
                        # Create LLM and memory
                        llm = create_llm(model_name, temperature, max_tokens)
                        
                        if llm:
                            try:
                                # Create conversation chain
                                conversation = ConversationChain(
                                    llm=llm,
                                    memory=st.session_state.memory,
                                    verbose=False
                                )
                                
                                # Add system prompt context
                                full_input = f"System: {st.session_state.system_prompt}\n\nUser: {last_user_message}"
                                
                                # Get new AI response
                                with st.spinner("🔄 Regenerating response..."):
                                    new_ai_response = conversation.predict(input=full_input)
                                
                                # Add new AI response to history
                                st.session_state.history.append(("assistant", new_ai_response))
                                st.session_state.chat_sessions[st.session_state.current_session].append(("assistant", new_ai_response))
                                
                                st.rerun()
                                
                            except Exception as e:
                                st.error(f"Error regenerating response: {str(e)}")
                                # Add error message as fallback
                                error_response = "I apologize, but I encountered an error while regenerating the response. Please try again."
                                st.session_state.history.append(("assistant", error_response))
                                st.session_state.chat_sessions[st.session_state.current_session].append(("assistant", error_response))
                else:
                    st.warning("No assistant response to regenerate!")
        
        with col2:
            if st.button("📝 Summarize Conversation"):
                if len(st.session_state.history) > 0:
                    # Create LLM
                    llm = create_llm(model_name, temperature, max_tokens)
                    
                    if llm:
                        try:
                            # Create conversation chain
                            conversation = ConversationChain(
                                llm=llm,
                                memory=st.session_state.memory,
                                verbose=False
                            )
                            
                            # Create summary prompt
                            summary_prompt = "Please provide a concise summary of our entire conversation so far, highlighting the main topics discussed and key points covered."
                            
                            # Add system prompt context
                            full_input = f"System: {st.session_state.system_prompt}\n\nUser: {summary_prompt}"
                            
                            # Get summary response
                            with st.spinner("📝 Generating conversation summary..."):
                                summary_response = conversation.predict(input=full_input)
                            
                            # Add summary request and response to history
                            st.session_state.history.append(("user", "📝 Summarize our conversation"))
                            st.session_state.history.append(("assistant", summary_response))
                            
                            # Add to chat sessions
                            st.session_state.chat_sessions[st.session_state.current_session].append(("user", "📝 Summarize our conversation"))
                            st.session_state.chat_sessions[st.session_state.current_session].append(("assistant", summary_response))
                            
                            st.rerun()
                            
                        except Exception as e:
                            st.error(f"Error generating summary: {str(e)}")
                            # Add error message as fallback
                            error_response = "I apologize, but I encountered an error while generating the conversation summary. Please try again."
                            st.session_state.history.append(("user", "📝 Summarize our conversation"))
                            st.session_state.history.append(("assistant", error_response))
                            st.session_state.chat_sessions[st.session_state.current_session].append(("user", "📝 Summarize our conversation"))
                            st.session_state.chat_sessions[st.session_state.current_session].append(("assistant", error_response))
                else:
                    st.warning("No conversation to summarize!")
        
        with col3:
            if st.button("🗑️ Clear Current Session"):
                st.session_state.history = []
                st.session_state.chat_sessions[st.session_state.current_session] = []
                st.session_state.memory = ConversationBufferMemory(return_messages=True)
                st.rerun()

if __name__ == "__main__":
    main()