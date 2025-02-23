import openai
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, Tool, AgentType
import sqlite3
import api
import api.app

# 设置 OpenAI API 密钥
openai.api_key = "your-openai-api-key"

class QAAgent:
    def __init__(self, model_name="text-davinci-003", temperature=0.7, db_name=":memory:"):
        self.conn = sqlite3.connect(db_name)
        self.c = self.conn.cursor()
        # self._create_database()

        self.prompt_template = """
        You are a helpful assistant. Based on the following input, provide a detailed and clear answer.

        Input: {input_text}
        Answer:
        """
        self.prompt = PromptTemplate(input_variables=["input_text"], template=self.prompt_template)
        
        # 初始化 
        self.llm = OpenAI(model=model_name, temperature=temperature)
        self.llm_chain = LLMChain(prompt=self.prompt, llm=self.llm)
        
        # 初始化内存 oom caution!!!
        self.memory = ConversationBufferMemory(memory_key="conversation_history", return_messages=True, max_memory=5)

        self.database_tool = DatabaseTool(self.conn)

        # 设置 Agent
        self.tools = [
            Tool(
                name="DatabaseTool",
                func=self.database_tool.save_to_database,
                description="Stores the question and answer pair in the database."
            ),
            Tool(
                name="DatabaseQueryTool",
                func=self.database_tool.query_database,
                description="Queries the database for an answer to a question."
            )
        ]
        self.agent = initialize_agent(
            self.tools,
            self.llm,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            memory=self.memory
        )

    def _create_database(self):
        self.c.execute('''CREATE TABLE IF NOT EXISTS qa_pairs (question TEXT, answer TEXT)''')
        self.conn.commit()

    def handle_user_input(self, user_input):
        """
        处理用户输入：首先查找数据库中的答案，如果没有找到，则使用 Agent 生成答案并保存到数据库
        """
    
        existing_answer = self.database_tool.query_database(user_input)
        
        if existing_answer:
            return f"Found in database: {existing_answer}"
        
        # response = self.agent.run(user_input)
        config = {
            "max_length": 50,
            "num_return_sequences": 1
        }
        response = api.app.lm.generate_response(user_input, **config)
        return response

    def get_all_qa_pairs(self):
        """ 获取数据库中的所有问答对 """
        self.c.execute("SELECT * FROM qa_pairs")
        return self.c.fetchall()

class DatabaseTool:
    def __init__(self, db_connection):
        self.db_connection = db_connection

    def save_to_database(self, question, answer):
        """ 保存问答对到数据库 """
        c = self.db_connection.cursor()
        c.execute('INSERT INTO qa_pairs (question, answer) VALUES (?, ?)', (question, answer))
        self.db_connection.commit()
        return "QA pair saved to database."

    def query_database(self, question):
        """ 查询数据库获取答案 """
        c = self.db_connection.cursor()
        c.execute('SELECT answer FROM qa_pairs WHERE question=?', (question,))
        result = c.fetchone()
        if result:
            return result[0]
        else:
            return None

# 示例使用
def test():
    # qa_agent = QAAgent(model_name="gpt-3.5-turbo", temperature=0.7)
    # user_input = "What is the capital of France?"
    # response = qa_agent.handle_user_input(user_input)
    user_input = "who r u"
    response = api.app.generate(user_input)
    print(response)  
    
    # 查看数据库中存储的问答对
    qa_agent = QAAgent(model_name="gpt-3.5-turbo", temperature=0.7)
    qa_pairs = qa_agent.get_all_qa_pairs()
    print(qa_pairs) 
