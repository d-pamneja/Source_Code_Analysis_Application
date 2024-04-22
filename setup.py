from setuptools import find_packages, setup 

setup(
    name="Source Code Analysis Application",
    version="0.0.1",
    author="Dhruv Pamneja",
    author_email="dpamneja@gmail.com",
    install_requires = ["openai","tiktoken","chromadb","langchain","python-dotenv","flask","GitPython"],
    packages=find_packages()
)