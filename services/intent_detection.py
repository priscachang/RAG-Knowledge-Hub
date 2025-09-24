from mistralai import Mistral

def detect_query_intent(query: str, llm_client: Mistral) -> str:
    """Enhanced intent detection using LLM."""
    try:
        intent_prompt = f"""Analyze the following user query and determine its intent. Return only one of these exact categories:

        Categories:
        - greeting: Simple greetings like "hi", "hello", "good morning"
        - question: Questions asking for information with words like what, how, why, when, where, who, which
        - list_request: Requests for lists, tables, or multiple items like "show me all", "list everything"
        - summary: Requests for summaries, overviews, or main points
        - finish: Goodbye messages like "thank you", "bye", "goodbye", "that's all", "I'm done"
        - general: Any other type of query

        User query: "{query}"

        Intent:"""

        response = llm_client.chat.complete(
            model="mistral-small-latest",
            messages=[{"role": "user", "content": intent_prompt}],
            temperature=0.1,
            max_tokens=10
        )
        
        intent = response.choices[0].message.content.strip().lower()
        
        valid_intents = ["greeting", "question", "list_request", "summary", "finish", "general"]
        if intent in valid_intents:
            return intent
        else:
            print(f"[DEBUG] LLM returned invalid intent: {intent}, defaulting to general")
            return "general"
            
    except Exception as e:
        print(f"[ERROR] LLM intent detection failed: {e}, using fallback")
        return _fallback_intent_detection(query)

def _fallback_intent_detection(query: str) -> str:
    """Fallback rule-based intent detection."""
    query_lower = query.lower()
    
    # Greeting 
    greetings = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]
    if any(greeting in query_lower for greeting in greetings):
        return "greeting"
    
    # Finish 
    finish_indicators = ["bye", "goodbye", "thank you", "thanks", "that's all", "i'm done", "finished", "done", "see you", "farewell"]
    if any(indicator in query_lower for indicator in finish_indicators):
        return "finish"
    
    # Question
    question_words = ["what", "how", "why", "when", "where", "who", "which"]
    if any(word in query_lower for word in question_words) or "?" in query:
        return "question"
    
    # List/table 
    list_indicators = ["list", "show me", "give me", "all", "every", "each"]
    if any(indicator in query_lower for indicator in list_indicators):
        return "list_request"
    
    # Summary
    summary_indicators = ["summary", "summarize", "overview", "main points"]
    if any(indicator in query_lower for indicator in summary_indicators):
        return "summary"
    
    return "general"

def enhance_query(query: str, intent: str) -> str:
    """Enhance query for better retrieval."""
    if intent == "list_request":
        return f"List all items related to: {query}"
    elif intent == "summary":
        return f"Provide a comprehensive summary of: {query}"
    elif intent == "question":
        return query
    else:
        return f"Information about: {query}"
