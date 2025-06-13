import os
import dotenv
from time import time
import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import json
import sys
from io import StringIO
import traceback
import matplotlib.pyplot as plt
import seaborn as sns
from contextlib import redirect_stdout, redirect_stderr

from langchain_community.document_loaders.text import TextLoader
from langchain_community.document_loaders import (
    CSVLoader,
    WebBaseLoader, 
    PyPDFLoader
)
# pip install docx2txt, pypdf
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, AzureOpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.tools import Tool
from langchain.agents import create_openai_tools_agent, AgentExecutor

dotenv.load_dotenv()

os.environ["USER_AGENT"] = "myagent"
DB_DOCS_LIMIT = 10

# ================================
# PYTHON/PANDAS CODE EXECUTION TOOL
# ================================

class PythonDataAnalyzer:
    """Tool for generating and executing Python/Pandas code for data analysis"""
    
    def __init__(self):
        self.available_data = {}
        self.execution_history = []
    
    def register_dataframe(self, name: str, df: pd.DataFrame):
        """Register a dataframe for analysis"""
        self.available_data[name] = df
        return f"DataFrame '{name}' registered with shape {df.shape}"
    
    def get_dataframe_info(self, df_name: str = None) -> str:
        """Get information about available dataframes"""
        if df_name and df_name in self.available_data:
            df = self.available_data[df_name]
            info = {
                "name": df_name,
                "shape": df.shape,
                "columns": df.columns.tolist(),
                "dtypes": df.dtypes.to_dict(),
                "sample_data": df.head(3).to_dict()
            }
            return json.dumps(info, indent=2, default=str)
        else:
            info = {}
            for name, df in self.available_data.items():
                info[name] = {
                    "shape": df.shape,
                    "columns": df.columns.tolist(),
                    "dtypes": df.dtypes.to_dict()
                }
            return json.dumps(info, indent=2, default=str)
    
    def execute_python_code(self, code: str, return_result: bool = True) -> Dict[str, Any]:
        """Execute Python/Pandas code safely and return results"""
        
        # Prepare execution environment
        exec_globals = {
            'pd': pd,
            'np': np,
            'plt': plt,
            'sns': sns,
            'json': json,
            # Add dataframes to globals
            **self.available_data
        }
        
        exec_locals = {}
        
        # Capture output
        output_buffer = StringIO()
        error_buffer = StringIO()
        
        try:
            # Execute the code
            with redirect_stdout(output_buffer), redirect_stderr(error_buffer):
                exec(code, exec_globals, exec_locals)
            
            # Get output
            stdout_output = output_buffer.getvalue()
            stderr_output = error_buffer.getvalue()
            
            # Find the result variable if return_result is True
            result_value = None
            if return_result:
                # Look for common result variable names
                for var_name in ['result', 'answer', 'output', 'final_result']:
                    if var_name in exec_locals:
                        result_value = exec_locals[var_name]
                        break
                
                # If no explicit result variable, try to get the last expression
                if result_value is None and exec_locals:
                    # Get the last assigned variable
                    last_var = list(exec_locals.keys())[-1]
                    result_value = exec_locals[last_var]
            
            execution_result = {
                "success": True,
                "result": result_value,
                "stdout": stdout_output,
                "stderr": stderr_output,
                "code_executed": code,
                "variables_created": list(exec_locals.keys())
            }
            
            # Store execution history
            self.execution_history.append(execution_result)
            
            return execution_result
            
        except Exception as e:
            error_result = {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__,
                "traceback": traceback.format_exc(),
                "code_executed": code,
                "stdout": output_buffer.getvalue(),
                "stderr": error_buffer.getvalue()
            }
            
            self.execution_history.append(error_result)
            return error_result
    
    def create_sample_data_from_context(self, context_text: str) -> pd.DataFrame:
        """Extract numerical data from context and create a sample DataFrame"""
        # This is a simplified extraction - in practice, you'd want more sophisticated parsing
        try:
            lines = context_text.split('\n')
            data_rows = []
            
            for line in lines:
                # Look for lines that might contain tabular data
                if ',' in line or '\t' in line:
                    # Try to extract numerical values
                    parts = line.replace('\t', ',').split(',')
                    numeric_parts = []
                    for part in parts:
                        try:
                            numeric_parts.append(float(part.strip()))
                        except:
                            numeric_parts.append(part.strip())
                    
                    if len(numeric_parts) > 1:
                        data_rows.append(numeric_parts)
            
            if data_rows:
                # Create DataFrame with generic column names
                max_cols = max(len(row) for row in data_rows)
                columns = [f'col_{i}' for i in range(max_cols)]
                
                # Pad rows to same length
                padded_rows = []
                for row in data_rows:
                    padded_row = row + [None] * (max_cols - len(row))
                    padded_rows.append(padded_row)
                
                df = pd.DataFrame(padded_rows, columns=columns)
                return df
            
        except Exception as e:
            print(f"Error creating sample data: {e}")
        
        # Return empty DataFrame if extraction fails
        return pd.DataFrame()

# Initialize analyzer
python_analyzer = PythonDataAnalyzer()

# ================================
# LANGCHAIN TOOLS INTEGRATION
# ================================

def create_python_analysis_tools():
    """Create LangChain tools for Python/Pandas analysis"""
    
    def python_code_executor(code_input: str) -> str:
        """
        Execute Python/Pandas code for data analysis. 
        Input should be valid Python code using pandas (pd), numpy (np), matplotlib (plt).
        Available dataframes: Use python_analyzer.available_data.keys() to see available data.
        Always assign your final result to a variable named 'result' for proper return.
        
        Example:
        ```python
        # Calculate growth rate
        values = [100, 110, 125, 140]
        df = pd.DataFrame({'values': values})
        growth_rates = df['values'].pct_change() * 100
        result = {
            'growth_rates': growth_rates.tolist(),
            'average_growth': growth_rates.mean(),
            'total_growth': ((values[-1] - values[0]) / values[0]) * 100
        }
        ```
        """
        try:
            # Clean the code input
            code = code_input.strip()
            if code.startswith('```python'):
                code = code.replace('```python', '').replace('```', '').strip()
            
            # Execute the code
            execution_result = python_analyzer.execute_python_code(code)
            
            if execution_result["success"]:
                response = {
                    "success": True,
                    "result": execution_result["result"],
                    "output": execution_result["stdout"],
                    "code_executed": execution_result["code_executed"]
                }
                
                # Convert result to JSON-serializable format
                if hasattr(execution_result["result"], 'to_dict'):
                    response["result"] = execution_result["result"].to_dict()
                elif hasattr(execution_result["result"], 'tolist'):
                    response["result"] = execution_result["result"].tolist()
                
                return json.dumps(response, indent=2, default=str)
            else:
                return json.dumps({
                    "success": False,
                    "error": execution_result["error"],
                    "traceback": execution_result["traceback"]
                }, indent=2)
                
        except Exception as e:
            return json.dumps({
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__
            }, indent=2)
    
    def dataframe_info_tool(df_name: str = "") -> str:
        """Get information about available dataframes for analysis"""
        return python_analyzer.get_dataframe_info(df_name.strip() if df_name else None)
    
    def create_dataframe_tool(data_input: str) -> str:
        """
        Create a DataFrame from structured data input.
        Input format: 'name:dataframe_name|data:csv_format_data'
        
        Example: 'name:coffee_data|data:Country,Year,Arabica,Robusta\\nBrazil,2020,100,50\\nColombia,2020,80,20'
        """
        try:
            parts = data_input.split('|')
            name_part = next((p for p in parts if p.startswith('name:')), None)
            data_part = next((p for p in parts if p.startswith('data:')), None)
            
            if not name_part or not data_part:
                return "Error: Input must contain 'name:dataframe_name' and 'data:csv_format_data'"
            
            df_name = name_part.replace('name:', '').strip()
            csv_data = data_part.replace('data:', '').strip()
            
            # Parse CSV data
            from io import StringIO
            df = pd.read_csv(StringIO(csv_data))
            
            # Register the dataframe
            result = python_analyzer.register_dataframe(df_name, df)
            
            return json.dumps({
                "success": True,
                "message": result,
                "dataframe_info": {
                    "name": df_name,
                    "shape": df.shape,
                    "columns": df.columns.tolist(),
                    "sample": df.head(3).to_dict()
                }
            }, indent=2, default=str)
            
        except Exception as e:
            return json.dumps({
                "success": False,
                "error": str(e)
            }, indent=2)
    
    tools = [
        Tool(
            name="execute_python_analysis",
            description="Execute Python/Pandas code for data analysis. Use this for all mathematical calculations, statistical analysis, data manipulation, and visualizations. Always assign your final result to a variable named 'result'.",
            func=python_code_executor
        ),
        Tool(
            name="get_dataframe_info",
            description="Get information about available dataframes including columns, data types, and sample data. Use this before writing analysis code.",
            func=dataframe_info_tool
        ),
        Tool(
            name="create_dataframe",
            description="Create a new DataFrame from structured data for analysis. Use when you need to work with specific data extracted from the context.",
            func=create_dataframe_tool
        )
    ]
    
    return tools

# Function to stream the response of the LLM 
def stream_llm_response(llm_stream, messages):
    response_message = ""

    for chunk in llm_stream.stream(messages):
        response_message += chunk.content
        yield chunk

    st.session_state.messages.append({"role": "assistant", "content": response_message})


# --- Indexing Phase ---

def load_doc_to_db():
    # Use loader according to doc type
    if "rag_docs" in st.session_state and st.session_state.rag_docs:
        docs = [] 
        for doc_file in st.session_state.rag_docs:
            if doc_file.name not in st.session_state.rag_sources:
                if len(st.session_state.rag_sources) < DB_DOCS_LIMIT:
                    os.makedirs("source_files", exist_ok=True)
                    file_path = f"./source_files/{doc_file.name}"
                    with open(file_path, "wb") as file:
                        file.write(doc_file.read())

                    try:
                        # Load CSV and register as DataFrame for analysis
                        loader = CSVLoader(file_path)
                        docs.extend(loader.load())
                        st.session_state.rag_sources.append(doc_file.name)
                        
                        # Also load as DataFrame for Python analysis
                        if doc_file.name.endswith('.csv'):
                            df = pd.read_csv(file_path)
                            df_name = doc_file.name.replace('.csv', '').replace(' ', '_')
                            python_analyzer.register_dataframe(df_name, df)

                    except Exception as e:
                        st.toast(f"Error loading document {doc_file.name}: {e}", icon="⚠️")
                        print(f"Error loading document {doc_file.name}: {e}")
                    
                    finally:
                        os.remove(file_path)

                else:
                    st.error(F"Maximum number of documents reached ({DB_DOCS_LIMIT}).")

        if docs:
            _split_and_load_docs(docs)
            st.toast(f"Document *{str([doc_file.name for doc_file in st.session_state.rag_docs])[1:-1]}* loaded successfully.", icon="✅")


def load_url_to_db():
    if "rag_url" in st.session_state and st.session_state.rag_url:
        url = st.session_state.rag_url
        docs = []
        if url not in st.session_state.rag_sources:
            if len(st.session_state.rag_sources) < 10:
                try:
                    loader = WebBaseLoader(url)
                    docs.extend(loader.load())
                    st.session_state.rag_sources.append(url)

                except Exception as e:
                    st.error(f"Error loading document from {url}: {e}")

                if docs:
                    _split_and_load_docs(docs)
                    st.toast(f"Document from URL *{url}* loaded successfully.", icon="✅")

            else:
                st.error("Maximum number of documents reached (10).")


def initialize_vector_db(docs):
    if "AZ_OPENAI_API_KEY" not in os.environ:
        embedding = OpenAIEmbeddings(api_key=st.session_state.openai_api_key)
    else:
        embedding = AzureOpenAIEmbeddings(
            api_key=os.getenv("AZ_OPENAI_API_KEY"), 
            azure_endpoint=os.getenv("AZ_OPENAI_ENDPOINT"),
            model="text-embedding-3-large",
            openai_api_version="2024-12-01-preview",
        )

    vector_db = Chroma.from_documents(
        documents=docs,
        embedding=embedding,
        collection_name=f"{str(time()).replace('.', '')[:14]}_" + st.session_state['session_id'],
    )

    # We need to manage the number of collections that we have in memory, we will keep the last 20
    chroma_client = vector_db._client
    collection_names = sorted([collection.name for collection in chroma_client.list_collections()])
    print("Number of collections:", len(collection_names))
    while len(collection_names) > 20:
        chroma_client.delete_collection(collection_names[0])
        collection_names.pop(0)

    return vector_db


def _split_and_load_docs(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=5000,
        chunk_overlap=1000,
    )

    document_chunks = text_splitter.split_documents(docs)

    if "vector_db" not in st.session_state:
        st.session_state.vector_db = initialize_vector_db(docs)
    else:
        st.session_state.vector_db.add_documents(document_chunks)


# --- Enhanced RAG with Python Analysis Tools ---

def _get_context_retriever_chain(vector_db, llm):
    retriever = vector_db.as_retriever()
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="messages"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation, focusing on the most recent messages."),
    ])
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)

    return retriever_chain


def get_conversational_rag_chain_with_python_tools(llm):
    """Enhanced RAG chain with Python/Pandas analysis tools"""
    
    # Create Python analysis tools
    tools = create_python_analysis_tools()
    
    # Enhanced system prompt that includes Python tool usage instructions
    system_prompt = """You are an expert Data Analytics Assistant for "High Garden Coffee,"
    an international coffee exporter. Your primary role is to empower internal teams (Sales, Marketing, Logistics) 
    with agile, data-driven decision-making by providing interactive access to insights derived from our advanced analytical solutions.

    **Your Core Function:**
    Leverage the outputs from our developed machine learning models and historical data to answer specific business queries related to domestic coffee consumption. 
    These models have analyzed historical data from 1990-2020 across various countries and coffee types.

    Countries: ['Angola', 'Bolivia (Plurinational State of)', 'Brazil', 'Burundi',
    'Ecuador', 'Indonesia', 'Madagascar', 'Malawi', 'Papua New Guinea',
    'Paraguay', 'Peru', 'Rwanda', 'Timor-Leste', 'Zimbabwe', 'Congo',
    'Cuba', 'Dominican Republic', 'Haiti', 'Philippines', 'Tanzania',
    'Zambia', 'Cameroon', 'Central African Republic', 'Colombia',
    'Costa Rica', "Côte d'Ivoire", 'Democratic Republic of Congo',
    'El Salvador', 'Equatorial Guinea', 'Ethiopia', 'Gabon', 'Ghana',
    'Guatemala', 'Guinea', 'Guyana', 'Honduras', 'India', 'Jamaica',
    'Kenya', "Lao People's Democratic Republic", 'Liberia', 'Mexico',
    'Nepal', 'Nicaragua', 'Nigeria', 'Panama', 'Sierra Leone',
    'Sri Lanka', 'Thailand', 'Togo', 'Trinidad & Tobago', 'Uganda',
    'Venezuela', 'Viet Nam', 'Yemen']

    Coffee Types: ['Mixta' 'Arabica' 'Robusta']

    **CRITICAL - Use Python/Pandas Tools for ALL Analysis:**
    You have access to powerful Python/Pandas code execution tools. You MUST use these tools for ANY mathematical calculation, statistical analysis, or data manipulation. 

    **Available Tools:**
    1. **execute_python_analysis**: Execute Python/Pandas code for calculations and analysis
    2. **get_dataframe_info**: Get information about available dataframes
    3. **create_dataframe**: Create new dataframes from context data

    **Python Tool Usage Guidelines:**

    1. **Always Use Python for Calculations**: NEVER perform manual calculations. Always write Python code.

    2. **Code Structure**: Always assign your final result to a variable named 'result':
    ```python
    # Your analysis code here
    result = final_answer_or_calculation
    ```

    3. **Data Extraction**: Extract numerical data from the retrieved context and create DataFrames:
    ```python
    # Extract data from context
    data = {{
        'Country': ['Brazil', 'Colombia', 'Peru'],
        'Arabica_2020': [1500, 800, 600],
        'Robusta_2020': [200, 100, 50]
    }}
    df = pd.DataFrame(data)
    
    # Perform your analysis
    result = df.groupby('Country').sum()
    ```

    4. **Common Analysis Patterns**:
    - **Growth Rates**: Use `pct_change()` or manual calculation
    - **Market Share**: Calculate percentages with proper totals
    - **Statistical Analysis**: Use pandas describe(), mean(), std(), etc.
    - **Forecasting**: Use simple linear regression or trend analysis
    - **Comparisons**: Use groupby(), pivot tables, and aggregations

    5. **Always Validate**: Check your results make sense before presenting them.

    **Example Python Usage:**
    ```python
    # Calculate growth rate for coffee consumption
    years = [2018, 2019, 2020]
    consumption = [1000, 1100, 1250]

    df = pd.DataFrame({{'Year': years, 'Consumption': consumption}})
    df['Growth_Rate'] = df['Consumption'].pct_change() * 100

    result = {{
        'average_growth_rate': df['Growth_Rate'].mean(),
        'total_growth_rate': ((consumption[-1] - consumption[0]) / consumption[0]) * 100,
        'yearly_rates': df['Growth_Rate'].dropna().tolist()
    }}
    ```

    **Data and Analytical Capabilities:**
    - **Predictive Analysis**: Forecasts for future coffee consumption volumes
    - **Supply Chain Optimization**: Demand forecasts and consumption ranges
    - **Market Segmentation**: International market segments analysis
    - **Innovation Opportunities**: Emerging trends and niches identification

    **Response Guidelines:**
    1. **Extract Data**: Always extract relevant numerical data from the retrieved context
    2. **Write Python Code**: Use Python tools for ALL calculations and analysis
    3. **Present Results**: Format the Python results in a business-friendly manner. Take into account that the data is in kg (not units), you can convert it to tons by dividing by 1000.
    4. **Accuracy First**: Never guess or estimate - always calculate
    5. **Context-Driven**: Base all analysis on the retrieved data context
    6. **Business Focus**: Frame results in terms of business implications for High Garden Coffee

    **Constraint:** Do not engage in general conversation outside the scope of "High Garden Coffee" data analytics. Always use Python tools for analysis. Maintain a professional, objective, and helpful tone.

    Context: {{context}}"""

    # Create the agent with tools
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])
    
    agent = create_openai_tools_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=5)
    
    return agent_executor


def stream_llm_rag_response_with_python_tools(llm_stream, messages):
    """Enhanced RAG response with Python/Pandas analysis tools"""
    
    # Get retriever for context
    retriever_chain = _get_context_retriever_chain(st.session_state.vector_db, llm_stream)
    
    # Get relevant context
    context_docs = retriever_chain.invoke({"messages": messages[:-1], "input": messages[-1].content})
    context = "\n\n".join([doc.page_content for doc in context_docs])
    
    # Create agent with Python tools
    agent_executor = get_conversational_rag_chain_with_python_tools(llm_stream)
    
    response_message = "*(RAG Response with Python Analysis)*\n"
    
    try:
        # Execute agent with context
        result = agent_executor.invoke({
            "messages": messages[:-1], 
            "input": messages[-1].content,
            "context": context
        })
        
        response_message += result["output"]
        yield result["output"]
        
    except Exception as e:
        error_msg = "He tenido inconveniente al intentar responder esta pregunta, puedes reformularla o hacerla más específica."#f"Error in Python analysis execution: {str(e)}"
        response_message += error_msg
        yield error_msg

    st.session_state.messages.append({"role": "assistant", "content": response_message})