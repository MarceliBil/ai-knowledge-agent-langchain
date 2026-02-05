import sys
from dotenv import load_dotenv

from rag.rag_chain import get_rag_chain


def main():
    load_dotenv()
    question = " ".join(sys.argv[1:])
    chain = get_rag_chain()
    answer = chain.invoke(question)
    print("\n" + answer + "\n")


if __name__ == "__main__":
    main()
