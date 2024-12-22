from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field


class ParaphrasedQuery(BaseModel):
    """You have performed query expansion to generate a paraphrasing of a question."""
    paraphrased_query: str = Field(
        ...,
        description="A unique paraphrasing of the original question.",
    )


class ParaphrasedQuerylist(BaseModel):
    """You have performed query expansion to generate a paraphrasing of a question."""
    content: list[ParaphrasedQuery] = Field()


class ExpansionQuery:
    def __init__(self, llm):
        self.parser = PydanticOutputParser(pydantic_object=ParaphrasedQuerylist)
        self.prompt = PromptTemplate(template="""Bạn là một chuyên gia trong việc chuyển đổi câu hỏi của người dùng thành các truy vấn cơ sở dữ liệu.
                                               Bạn có quyền truy cập vào cơ sở dữ liệu chứa các video hướng dẫn về một thư viện phần mềm dùng để 
                                                xây dựng ứng dụng dựa trên LLM.Hãy thực hiện việc mở rộng truy vấn. Nếu có nhiều cách diễn đạt phổ biến cho 
                                                câu hỏi của người dùng hoặc các từ đồng nghĩa phổ biến với những từ khóa trong câu hỏi, hãy đảm bảo trả về 
                                                hiều phiên bản truy vấn với các cách diễn đạt khác nhau.Nếu có các từ viết tắt hoặc từ mà bạn không quen thuộc, 
                                                đừng cố gắng diễn đạt lại chúng.Trả về ít nhất 5 phiên bản của câu hỏi.
                                                question:{query}
                                                
                                                {format_instructions}
                                                """,
                                     input_variables=["query"],
                                     partial_variables={"format_instructions": self.parser.get_format_instructions()}
                                     )
        self.llm = llm

    def invoke(self, query):
        prompt = self.prompt.format(query=query)
        result = self.llm.invoke(prompt)

        return self.parser.parse(result)
