from mistralai import Mistral

def get_prompt_template(intent: str, context: str, query: str) -> str:
    """Get appropriate prompt template based on intent."""
    
    if intent == "greeting":
        return "You are a helpful assistant. Respond to the greeting in a friendly manner."
    
    elif intent == "finish":
        return "You are a helpful assistant. The user is ending the conversation. Respond with a polite goodbye message, thanking them for using the service and wishing them well."
    
    elif intent == "list_request":
        return f"""You are a helpful assistant. Based on the following context, provide a structured list of items related to the query.

        Context:
        {context}

        Query: {query}

        Please provide a clear, organized list with bullet points or numbered items. If the information is not available in the context, say "The requested information is not available in the provided context."

        List:"""

    elif intent == "summary":
        return f"""You are a helpful assistant. Based on the following context, provide a comprehensive summary into 3 to 5 shortbullet points. Answer in plain text only. 

        Context:
        {context}

        Query: {query}

        Please provide a well-structured summary covering the main points. If the information is not available in the context, say "The requested information is not available in the provided context."

        Summary:"""

    else:  # general or question
        return f"""You are a helpful assistant. Use only the following context to answer the question. If the answer cannot be found in the context, say "The information is not available in the provided context."

        Context:
        {context}

        Question: {query}

        Answer:"""

def generate_answer(prompt: str, llm_client: Mistral) -> str:
    """Generate answer using LLM."""
    try:
        response = llm_client.chat.complete(
            model="mistral-small-latest",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"LLM generation error: {e}")
        raise Exception(f"LLM generation failed: {str(e)}")
