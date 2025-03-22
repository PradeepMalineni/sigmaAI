import streamlit as st
from transformers import AutoTokenizer, pipeline
import torch
from openai import OpenAI
from dotenv import load_dotenv
import os

# Load environment variables from the .env file
load_dotenv()

# Retrieve the API tokens from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HF_API_TOKEN = os.getenv("HF_API_TOKEN")

# Initialize OpenAI client using the API key
client = OpenAI(api_key=OPENAI_API_KEY)

# Cache the Hugging Face model to reduce loading times
@st.cache_resource
def load_hf_fallback_model():
    try:
        # Load the tokenizer and model from Hugging Face
        model = "tiiuae/falcon-7b-instruct"
        tokenizer = AutoTokenizer.from_pretrained(model, use_auth_token=HF_API_TOKEN)
        
        # Create the pipeline for text generation
        falcon_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            torch_dtype=torch.float16,  # Use float16 if supported
            trust_remote_code=True,
            device_map="auto",  # Automatically assign the model to available devices (GPU/CPU)
            # Pass generation arguments inside generation_kwargs
            generation_kwargs={
                "temperature": 0.3,  # Controls randomness of predictions
                "max_length": 768,   # Maximum number of tokens in the output
                "max_new_tokens": 768  # How many tokens to generate (should be inside generation_kwargs)
            }
        )
        
        return falcon_pipeline

    except Exception as e:
        print(f"Failed to load 'tiiuae/falcon-7b-instruct'. Falling back to T5. Error: {e}")
        # Fallback to a default model (e.g., t5-small)
        return pipeline(
            "text-generation",
            model="t5-small",  # Default fallback model
            use_auth_token=HF_API_TOKEN,
            model_kwargs={
                "temperature": 0.3,
                "max_new_tokens": 768
            },
            return_full_text=True
        )

hf_model = load_hf_fallback_model()

def analyze_with_fallback(query_text, similar_df):
    summary = "\n".join([
        f"Incident {row['incident_id']} with CI {row['ci_id']} had this issue: {row['description']}. "
        f"Resolution applied: {row['resolution']}. "
        f"Cause identified: {row['cause'] if row['cause'].strip().lower() != 'unknown' else 'not explicitly documented'}."
        for _, row in similar_df.iterrows()
    ])

    instruction_header = """
You are a platform SRE assistant. 
Analyze the reported issue and similar past incidents to infer the root cause and generate a detailed resolution plan.

🚨 Begin ONLY in the format below. No explanation or intro required.
---
**Probable Root Cause**:
<explanation>

**Resolution Plan**:
1. Step one...
2. Step two...

**Preventive Suggestions**:
- Tip 1
- Tip 2
---
"""

    few_shot_example = """
Example:

Issue:
"Backend API call timeout for App ABC"

Similar Incidents:
Incident INC2010: App unable to connect to backend | Resolution: Restarted backend pod | Cause: DNS resolution failure  
Incident INC2033: Backend API call failed intermittently | Resolution: Restarted service | Cause: Load balancer misconfiguration

---
**Probable Root Cause**:
Backend services were not reachable due to DNS resolution failures and misconfigured routing rules.

**Resolution Plan**:
1. Restart affected backend pods.
2. Flush and revalidate DNS cache entries.
3. Fix routing rules in load balancer configs.

**Preventive Suggestions**:
- Enable DNS monitoring and proactive alerts.
- Add failover DNS records.
- Validate load balancer configs on each deploy.
---
"""

    prompt = f"{instruction_header}\n\n{few_shot_example}\n\nIssue: \"{query_text}\"\n\nSimilar Incidents:\n{summary}\n\nNow respond below:\n"

    # Function to generate the response with retry logic
    def generate_response_with_retry(prompt):
        result = hf_model(prompt[:1500], max_length=768, do_sample=True, top_p=0.9)
        output = result[0]['generated_text'].strip()

        # If output is vague or lacks structure, try once more
        if len(output.split()) < 25 or "probable root cause" not in output.lower():
            retry_prompt = prompt + "\n\nYour last response was incomplete. Please generate a more detailed answer following the full format."
            result = hf_model(retry_prompt[:1500], max_length=768, do_sample=True, top_p=0.9)
            output = result[0]['generated_text'].strip()
            st.caption("🔄 Re-prompted model to improve detail.")

        return output

    # Primary: OpenAI API (fallback option)
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        return response.choices[0].message.content

    # Fallback: Hugging Face
    except Exception as e:
        print("OpenAI failed. Using Hugging Face fallback.")
        return generate_response_with_retry(prompt)
