from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import streamlit as st
import torch


faq = {
    "what is the ideal water–cement ratio for m25 grade concrete": 
        "The ideal water–cement ratio for M25 grade concrete is about 0.45 to 0.50.",
    
    "how many days should concrete be cured": 
        "Concrete should be cured for at least 7 days (OPC) or 10–14 days (PPC). For best results, 28 days curing is recommended.",
    
    "standard size of brick in india": 
        "The standard brick size in India is 190 × 90 × 90 mm (without mortar). With mortar, it becomes 200 × 100 × 100 mm.",
    
    "weight of 12mm dia steel per meter": 
        "The weight of 12 mm dia steel is about 0.89 kg per meter. Formula: W = d²/162.",
    
    "what is lap length in reinforcement": 
        "Lap length is the overlap length between two rebars to transfer stress. Usually 40d in tension and 30d in compression (d = bar diameter).",
    
    "difference between one way slab and two way slab": 
        "One-way slab bends in one direction (L/B > 2), while two-way slab bends in both directions (L/B < 2).",
    
    "minimum cover for slab reinforcement": 
        "As per IS 456, the minimum cover for slab reinforcement is 20 mm.",
    
    "difference between opc and ppc cement": 
        "OPC (Ordinary Portland Cement) gains strength faster, while PPC (Pozzolana Portland Cement) is more durable, economical, and resistant to chemicals.",
    
    "what is shuttering in construction": 
        "Shuttering is temporary formwork used to hold wet concrete until it gains sufficient strength.",
    
    "what are safety precautions on a construction site": 
        "Use PPE (helmets, gloves, boots), ensure proper scaffolding, fall protection, safe tool handling, fire safety, and first aid readiness.",
    
    "difference between english bond and flemish bond": 
        "English bond alternates courses of headers and stretchers. Flemish bond mixes headers and stretchers in the same course.",
    
    "is code for plain and reinforced concrete": 
        "The IS code for plain and reinforced concrete is IS 456:2000.",
    
    "is code for steel design": 
        "The IS code for steel design is IS 800:2007.",
    
    "what is scaffolding": 
        "Scaffolding is a temporary framework used to support workers and materials during construction, repair, or maintenance.",
    
    "what is curing in concrete": 
        "Curing is the process of maintaining adequate moisture, temperature, and time to allow concrete to achieve desired strength and durability."
}

model_name = "google/flan-t5-small"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

st.title("🏗️ Construction Chatbot")

user_input = st.text_input("Ask me something about construction:")

def is_construction_query(query):
    keywords = [
        "concrete","cement","beam","steel","column","IS code","slab",
        "foundation","brick","mortar","curing","rebar","construction","site","safety"
    ]
    return any(k in query.lower() for k in keywords)

if user_input:
    query = user_input.lower()

    
    if query in faq:
        st.write("👷 Answer:", faq[query])

    
    elif not is_construction_query(query):
        st.write("❌ I only answer construction-related queries.")

    
    else:
        inputs = tokenizer(query, return_tensors="pt")
        outputs = model.generate(**inputs, max_new_tokens=150)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        st.write("👷 Answer:", response)
