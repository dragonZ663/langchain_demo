from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
import os



def main():
    print("Hello from langchain-demo!")
    llm = ChatOpenAI(
        model="deepseek-coder-v2-lite-instruct",
        base_url=os.environ.get("LM_STUDIO_BASE_URL"),
        api_key=os.environ.get("LM_STUDIO_API_KEY")
    )

    response = llm.invoke(input="你好，你有哪些能力，例如：推理，工具调用，文本生成等？")
    print(response)


if __name__ == "__main__":
    main()
